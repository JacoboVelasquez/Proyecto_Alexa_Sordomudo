"""
SERVIDOR WEB — une los 4 temas en un solo flujo y los muestra en el navegador.

  cámara -> [1 calidad] -> [2 YOLO] -> [3 procesamiento de la ROI] -> [4 ML] -> [5 decisión]
         -> letra por USB -> Pico -> LED / ventilador / servo
         -> [6 agente] escribe en el chat qué se hizo (o qué no) y por qué; responde preguntas

Uso:
  python servidor_web.py                          # cámara 0, busca la Pico sola
  python servidor_web.py --camara 1 --puerto COM5
  python servidor_web.py --camara demo_val.mp4    # video de ejemplo (se repite en bucle)
  python servidor_web.py --simular                # sin Pico: simulador que imita main.cpp
Luego abrir  http://localhost:5000

Un solo proceso es dueño de la cámara y del puerto serial (Windows no deja abrirlos dos veces):
la visión, los botones y el agente mandan TODAS las órdenes por la misma función enviar().
"""
import argparse
import csv
import json
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import agente as agente_mod
import clasificador_ml
import procesamiento as proc
from control_senas import COMANDOS, CONF_MIN, FRAMES_CONFIRMAR, TIEMPO_LIBRE, Camara, abrir_pico

BASE = Path(__file__).resolve().parent
RUNS = BASE / "runs" / "senas"
ML_MIN = 0.50          # probabilidad mínima que el ML debe darle a la clase de YOLO (calibrada en evaluar_sistema.py)
CONF_VER = 0.25        # YOLO reporta cajas desde 0.25 para poder explicar "confianza baja"; acciona desde CONF_MIN
CUADROS_AVISO = 6      # cuadros seguidos viendo una seña rechazada antes de que el agente escriba "no lo hice"
CONF_AVISO = 0.40      # por debajo de esto ni se avisa (suele ser ruido, no una seña intencional)
SEG_AVISO_BLOQUEO = 2.5  # si la seña sigue arriba 2.5 s después de ejecutarla, el agente explica que debe bajar la mano

# ------------------------------------------------------------------ estado compartido
# Varios hilos (visión, lector serial, Flask) leen y escriben esto: siempre con el candado.
candado = threading.Lock()
estado = {
    "actuadores": {"led": False, "ventilador": False, "servo": False},
    "linea_pico": "",
    "conexion": "iniciando...",
    "simulado": True,
    "fps": 0.0,
    "dispositivo": "CPU",
    "fuente": "",
    "ml_disponible": False,
    "aviso": "",
    "parametros": {"conf_min": CONF_MIN, "frames_confirmar": FRAMES_CONFIRMAR, "ml_min": ML_MIN, "modo": "yolo+ml"},
    "confirmacion": {"sena": None, "cuenta": 0, "bloqueada": None},
}
registros = deque(maxlen=50)     # registro de decisión de los últimos cuadros
eventos = deque(maxlen=200)      # órdenes enviadas: {hora, origen, comando, estado_resultante}
jpeg_video = None                # último cuadro anotado (para /video)
jpeg_etapas = None               # última tira de etapas del procesamiento (para /etapas.jpg)
jpeg_roi = None                  # último recorte de la mano (ROI) que entregó YOLO (para /roi.jpg)
info_roi = {}                    # de qué cuadro salió esa ROI: hora, seña YOLO, predicción ML
CAPTURAS = BASE / "capturas"     # aquí se guardan las ROI que pide la persona o el agente
activo = True


def hora():
    return datetime.now().strftime("%H:%M:%S")


# ------------------------------------------------------------------ Pico real o simulada
class SimuladorPico:
    """Imita EXACTAMENTE pico_senas/src/main.cpp con la misma interfaz que serial.Serial
    (write, in_waiting, read, close), así el resto del código no sabe si es real o simulada."""

    def __init__(self, led=False, ventilador=False, servo=False):
        self.led, self.ventilador, self.servo = led, ventilador, servo
        self.salida = b""

    def write(self, datos):
        for c in datos.decode(errors="ignore").upper():
            if c == "L":
                self.led = not self.led
            elif c == "V":
                self.ventilador = not self.ventilador
            elif c == "S":
                self.servo = not self.servo
            elif c == "T":                                   # si algo está encendido apaga todo; si no, enciende todo
                alguno = self.led or self.ventilador or self.servo
                self.led = self.ventilador = self.servo = not alguno
            elif c != "?":
                continue                                      # ignora otros caracteres, igual que la Pico
            on = lambda b: "ON" if b else "OFF"
            self.salida += f"LED={on(self.led)} VENT={on(self.ventilador)} SERVO={on(self.servo)}\r\n".encode()

    @property
    def in_waiting(self):
        return len(self.salida)

    def read(self, n):
        datos, self.salida = self.salida[:n], self.salida[n:]
        return datos

    def close(self):
        pass


pico = None
candado_serial = threading.Lock()
respuesta = threading.Event()         # se activa cuando la Pico responde una línea de estado


def pasar_a_simulado(motivo):
    """Si la Pico falla en plena demo, se sigue con el simulador desde el último estado conocido."""
    global pico
    a = estado["actuadores"]
    pico = SimuladorPico(a["led"], a["ventilador"], a["servo"])
    with candado:
        estado["simulado"] = True
        estado["conexion"] = f"Modo simulado ({motivo})"
    print(f"[AVISO] {motivo}: sigo en modo simulado.")


def hilo_lector_serial():
    """Lee lo que responde la Pico y actualiza el estado. La Pico es la fuente de verdad."""
    buffer = ""
    patron = re.compile(r"LED=(ON|OFF)\s+VENT=(ON|OFF)\s+SERVO=(ON|OFF)")
    while activo:
        try:
            with candado_serial:
                n = pico.in_waiting
                datos = pico.read(n).decode(errors="ignore") if n else ""
        except Exception as e:                               # cable desconectado, etc.
            pasar_a_simulado(f"Pico desconectada: {e.__class__.__name__}")
            continue
        buffer += datos
        while "\n" in buffer:
            linea, buffer = buffer.split("\n", 1)
            m = patron.search(linea)
            if m:
                with candado:
                    estado["actuadores"] = {"led": m[1] == "ON", "ventilador": m[2] == "ON", "servo": m[3] == "ON"}
                    estado["linea_pico"] = linea.strip()
                respuesta.set()
        time.sleep(0.01)


def enviar(letra, origen, esperar=True):
    """ÚNICO punto por donde salen órdenes a la Pico (visión, botones y agente).
    Escribe la letra, espera la respuesta (hasta 0.5 s) y deja el evento en el historial."""
    letra = letra.upper()
    if letra not in "LVST?":
        raise ValueError(f"Letra inválida: {letra}")
    antes = dict(estado["actuadores"])
    with candado_serial:
        respuesta.clear()
        try:
            pico.write(letra.encode())
        except Exception as e:
            pasar_a_simulado(f"error al escribir: {e.__class__.__name__}")
            pico.write(letra.encode())
    if esperar:
        respuesta.wait(0.5)
    with candado:
        despues = dict(estado["actuadores"])
        linea = estado["linea_pico"]
    if letra != "?":
        eventos.append({"hora": hora(), "origen": origen, "comando": letra, "estado_resultante": linea})
    return antes, despues


# ------------------------------------------------------------------ etapas 1-4 de un cuadro
def analizar_cuadro(frame, yolo, params, dispositivo="cpu"):
    """Pasa UN cuadro por calidad -> YOLO -> procesamiento -> ML y decide si el cuadro es válido.
    Devuelve (registro, datos_para_dibujar). El registro dice el MOTIVO si no es válido.
    También lo usa evaluar_sistema.py, así las pruebas miden exactamente lo mismo que corre en vivo."""
    reg = {"hora": hora(), "modo": params["modo"], "calidad": None, "yolo": None, "procesamiento": None,
           "ml": None, "valido": False, "sena": None, "etapa_falla": None, "motivo": "", "enviado": None}
    dibujo = None

    # 1) Calidad del cuadro (procesamiento básico)
    nitidez, brillo, motivo = proc.calidad_cuadro(frame)
    reg["calidad"] = {"nitidez": round(nitidez), "brillo": round(brillo), "ok": motivo is None}
    if motivo:
        reg.update(etapa_falla="calidad", motivo=motivo)
        return reg, dibujo

    # 2) Deep Learning: YOLO busca la mano y dice qué seña es
    res = yolo.predict(frame, conf=CONF_VER, verbose=False, device=dispositivo)[0]
    if not len(res.boxes):
        reg.update(etapa_falla="yolo", motivo="sin mano")
        return reg, dibujo
    b = res.boxes[int(res.boxes.conf.argmax())]               # la caja más segura
    clase, conf = res.names[int(b.cls[0])], float(b.conf[0])
    caja = [int(v) for v in b.xyxy[0].tolist()]
    reg["yolo"] = {"clase": clase, "conf": round(conf, 3), "caja": caja}

    # 3) Procesamiento de la ROI de la mano: segmentación, contorno y características
    roi = proc.recortar_roi(frame, caja)
    vector, (roi_m, mascara, contorno) = proc.procesar_roi(roi) if roi is not None else (None, (None, None, None))
    if vector is not None:
        reg["procesamiento"] = {"contorno": True, "defectos": int(vector[5]), "solidez": round(float(vector[3]), 3),
                                "circularidad": round(float(vector[4]), 3), "aspecto": round(float(vector[1]), 3),
                                "area": round(float(vector[0]), 3)}
    else:
        reg["procesamiento"] = {"contorno": False}

    # 4) Machine Learning: el clasificador clásico opina con esas características
    if vector is not None and estado["ml_disponible"]:
        pred, probs = clasificador_ml.predecir(vector)
        reg["ml"] = {"clase": pred, "prob": round(probs[pred], 3), "prob_yolo": round(probs.get(clase, 0), 3),
                     "probs": {k: round(v, 3) for k, v in probs.items()}}
    if roi_m is not None:
        dibujo = (roi_m, mascara, contorno, vector, (reg["ml"]["clase"], reg["ml"]["prob"]) if reg["ml"] else None)

    # 5) ¿El cuadro es válido?
    if conf < params["conf_min"]:
        reg.update(etapa_falla="yolo", motivo=f"confianza baja {conf:.2f} < {params['conf_min']:.2f}")
    elif params["modo"] == "yolo+ml" and vector is None:
        reg.update(etapa_falla="procesamiento", motivo="no se encontró el contorno de la mano")
    elif params["modo"] == "yolo+ml" and reg["ml"] is None:
        reg.update(etapa_falla="ml", motivo="clasificador ML no disponible")
    elif params["modo"] == "yolo+ml" and reg["ml"]["prob_yolo"] < params["ml_min"]:
        ml = reg["ml"]
        reg.update(etapa_falla="ml", motivo=(f"ML no coincide: predijo {ml['clase']} {ml['prob']:.2f}"
                                             if ml["clase"] != clase else
                                             f"ML poco seguro: {ml['prob_yolo']:.2f} < {params['ml_min']:.2f}"))
    else:
        reg.update(valido=True, sena=clase, motivo="seña válida")
    return reg, dibujo


class Confirmador:
    """Etapa 5: la misma seña válida en FRAMES_CONFIRMAR cuadros seguidos = orden confirmada.
    Anti-rebote: después de enviar, esa seña queda 'bloqueada' hasta que la mano baje TIEMPO_LIBRE s
    (así no se alterna 20 veces por dejar la mano quieta). Una seña distinta se acepta enseguida.
    También decide cuándo el agente debe escribir 'no lo hice y por qué' (una vez por intento)."""

    def __init__(self):
        self.candidata, self.cuenta = None, 0
        self.bloqueada, self.t_visto, self.t_envio = None, 0.0, 0.0
        self.aviso_bloqueo = False
        self.rech_clase, self.rech_cuenta, self.rech_avisado = None, 0, False

    def paso(self, reg, params, ahora):
        """Actualiza la cuenta con el registro del cuadro. Devuelve (letra_a_enviar | None, aviso | None)."""
        sena = reg["sena"]
        y = reg["yolo"]
        mano_arriba = y is not None and y["conf"] >= params["conf_min"]
        self.cuenta = self.cuenta + 1 if sena and sena == self.candidata else (1 if sena else 0)
        self.candidata = sena
        if mano_arriba:
            self.t_visto = ahora
        elif ahora - self.t_visto > TIEMPO_LIBRE:
            self.bloqueada, self.aviso_bloqueo = None, False      # mano abajo: ya se puede repetir
            self.rech_avisado, self.rech_cuenta = False, 0

        aviso = self._revisar_rechazo(reg)
        letra = None
        if sena and sena == self.bloqueada:
            reg["motivo"] = "bloqueada: baja la mano para repetir"
            if not self.aviso_bloqueo and ahora - self.t_envio > SEG_AVISO_BLOQUEO:
                self.aviso_bloqueo = True
                aviso = ("bloqueada", reg)
        elif sena and self.cuenta < params["frames_confirmar"]:
            reg["motivo"] = f"confirmando {self.cuenta}/{params['frames_confirmar']}"
        elif sena:
            self.bloqueada, self.aviso_bloqueo, self.t_envio = sena, False, ahora
            letra = COMANDOS[sena]
            reg["motivo"], reg["enviado"] = f"ENVIADO {letra}", letra
        with candado:
            estado["confirmacion"] = {"sena": sena, "cuenta": self.cuenta, "bloqueada": self.bloqueada}
        return letra, aviso

    def _revisar_rechazo(self, reg):
        """Si YOLO ve una seña pero el cuadro se rechaza varios cuadros seguidos, avisar UNA vez."""
        y = reg["yolo"]
        if reg["valido"] or y is None or y["conf"] < CONF_AVISO:
            if reg["valido"]:
                self.rech_cuenta = 0
            return None
        if y["clase"] != self.rech_clase:
            self.rech_clase, self.rech_cuenta, self.rech_avisado = y["clase"], 0, False
        self.rech_cuenta += 1
        if self.rech_cuenta >= CUADROS_AVISO and not self.rech_avisado:
            self.rech_avisado = True
            return ("rechazo", dict(reg))
        return None


# ------------------------------------------------------------------ hilo de visión
def dibujar_cuadro(frame, reg, fps):
    """Caja de YOLO + predicción ML + motivo, sobre el cuadro (lo que se ve en /video)."""
    y = reg["yolo"]
    color = (0, 200, 0) if reg["valido"] else (0, 0, 230)
    if y:
        x1, y1, x2, y2 = y["caja"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        texto = f"YOLO {y['clase']} {y['conf']:.2f}"
        if reg["ml"]:
            texto += f" | ML {reg['ml']['clase']} {reg['ml']['prob']:.2f}"
        cv2.putText(frame, texto, (x1, max(40, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    alto = frame.shape[0]
    cv2.rectangle(frame, (0, alto - 60), (frame.shape[1], alto), (0, 0, 0), -1)
    cv2.putText(frame, reg["motivo"], (15, alto - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    cv2.putText(frame, f"{fps:4.1f} FPS", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    return frame


def cuadro_mensaje(texto):
    img = np.zeros((480, 854, 3), np.uint8)
    cv2.putText(img, texto, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    return img


def hilo_vision(fuente, ruta_modelo, agente):
    global jpeg_video, jpeg_etapas, jpeg_roi, info_roi
    import torch
    from ultralytics import YOLO

    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    with candado:
        estado["dispositivo"] = f"GPU ({torch.cuda.get_device_name(0)})" if dispositivo == 0 else "CPU"
    yolo = YOLO(ruta_modelo)
    confirmador = Confirmador()
    cam, fps, t_prev, fallos = None, 0.0, time.time(), 0

    while activo:
        # --- abrir la cámara (y reintentar si no abre, sin tumbar el servidor) ---
        if cam is None:
            try:
                cam = Camara(fuente)
                pausa_video = 1 / (cam.cap.get(cv2.CAP_PROP_FPS) or 10) if cam.es_video else 0
                with candado:
                    estado["fuente"] = f"video {fuente}" if cam.es_video else f"cámara {fuente}"
                    estado["aviso"] = ""
            except SystemExit as e:
                with candado:
                    estado["aviso"] = f"{e} — reintento en 3 s"
                jpeg_video = cv2.imencode(".jpg", cuadro_mensaje(str(e)))[1].tobytes()
                time.sleep(3)
                continue

        t0 = time.time()
        ok, frame = cam.leer()
        if not ok or frame is None or frame.size == 0:          # cuadro vacío
            if cam.es_video:
                cam.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)          # el video de demo se repite en bucle
            else:
                fallos += 1
                if fallos > 300:                                 # ~3 s sin imagen: cámara desconectada, reabrir
                    with candado:
                        estado["aviso"] = "La cámara dejó de enviar imagen: reconectando…"
                    cam.cerrar()
                    cam, fallos = None, 0
                time.sleep(0.01)
            continue
        fallos = 0

        with candado:
            params = dict(estado["parametros"])
        reg, dibujo = analizar_cuadro(frame, yolo, params, dispositivo)
        letra, aviso = confirmador.paso(reg, params, time.time())

        if letra:                                                # 6) orden confirmada -> Pico
            antes, despues = enviar(letra, "seña")
            agente.avisar_hecho(letra, "seña", antes, despues, reg, params)
        elif aviso:
            agente.avisar_no_hecho(aviso[0], aviso[1], params)

        ahora = time.time()
        fps = 0.9 * fps + 0.1 / max(1e-6, ahora - t_prev)
        t_prev = ahora
        registros.append(reg)
        with candado:
            estado["fps"] = round(fps, 1)

        if reg["yolo"]:                                          # ROI de la mano: se recorta ANTES de dibujar encima
            roi = proc.recortar_roi(frame, reg["yolo"]["caja"])
            if roi is not None:
                escala_roi = min(1.0, 360 / max(roi.shape[:2]))     # máximo 360 px de lado para la web
                roi = cv2.resize(roi, (int(roi.shape[1] * escala_roi), int(roi.shape[0] * escala_roi)))
                jpeg_roi = cv2.imencode(".jpg", roi)[1].tobytes()
                info_roi = {"t": time.time(), "hora": reg["hora"], "yolo": reg["yolo"], "ml": reg["ml"]}

        vista = dibujar_cuadro(frame, reg, fps)
        escala = 960 / vista.shape[1]
        vista = cv2.resize(vista, (960, int(vista.shape[0] * escala)))
        jpeg_video = cv2.imencode(".jpg", vista, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
        if dibujo:
            jpeg_etapas = cv2.imencode(".jpg", proc.dibujar_etapas(*dibujo))[1].tobytes()

        if cam.es_video:                                         # el video va a su velocidad real
            time.sleep(max(0, pausa_video - (time.time() - t0)))
    if cam:
        cam.cerrar()


# ------------------------------------------------------------------ métricas para la web y el agente
def leer_json(ruta):
    try:
        return json.loads(Path(ruta).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def metricas():
    """YOLO (de results.csv), ML (metricas_ml.json), pruebas (pruebas.json) y conteo del dataset."""
    yolo = None
    try:
        with open(RUNS / "results.csv", encoding="utf-8") as f:
            filas = [{k.strip(): float(v) for k, v in fila.items()} for fila in csv.DictReader(f)]
        resumen = lambda r: {"epoca": int(r["epoch"]), "precision": round(r["metrics/precision(B)"], 3),
                             "recall": round(r["metrics/recall(B)"], 3), "map50": round(r["metrics/mAP50(B)"], 3),
                             "map50_95": round(r["metrics/mAP50-95(B)"], 3)}
        # Ultralytics guarda best.pt en la época con mejor "fitness" = 0.1*mAP50 + 0.9*mAP50-95
        mejor = max(filas, key=lambda r: 0.1 * r["metrics/mAP50(B)"] + 0.9 * r["metrics/mAP50-95(B)"])
        yolo = {"mejor": resumen(mejor), "ultima": resumen(filas[-1]), "epocas": len(filas)}
    except (OSError, KeyError, ValueError):
        pass
    conteo = {}
    for split in ("train", "val"):
        if not (BASE / "dataset_senas" / "labels" / split).exists():
            continue                                         # el dataset solo hace falta para reentrenar
        c = {k: 0 for k in COMANDOS}
        c["negativas"] = 0
        for lab in (BASE / "dataset_senas" / "labels" / split).glob("*.txt"):
            lineas = [l for l in lab.read_text().splitlines() if l.strip()]
            if not lineas:                                   # archivo vacío = foto negativa (sin seña)
                c["negativas"] += 1
            for l in lineas:                                 # primer número de cada línea = id de la clase
                c[clasificador_ml.CLASES[int(l.split()[0])]] += 1
        conteo[split] = c
    imagenes = [n for n in ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png",
                            "val_batch0_pred.jpg", "labels.jpg") if (RUNS / n).exists()]
    return {"yolo": yolo, "ml": leer_json(clasificador_ml.RUTA_METRICAS),
            "pruebas": leer_json(BASE / "modelos" / "pruebas.json"), "conteo": conteo or None,
            "imagenes": [f"/runs/{n}" for n in imagenes]}


def ajustar_parametros(cambios):
    """Cambia parámetros con límites (los usan la web y el agente). Devuelve (parametros, errores)."""
    errores = []
    with candado:
        p = estado["parametros"]
        if cambios.get("conf_min") is not None:
            p["conf_min"] = round(min(0.95, max(0.30, float(cambios["conf_min"]))), 2)
        if cambios.get("frames_confirmar") is not None:
            p["frames_confirmar"] = int(min(15, max(1, int(cambios["frames_confirmar"]))))
        if cambios.get("ml_min") is not None:
            p["ml_min"] = round(min(1.0, max(0.0, float(cambios["ml_min"]))), 2)
        modo = cambios.get("modo")
        if modo is not None:
            if modo not in ("yolo", "yolo+ml"):
                errores.append("modo debe ser 'yolo' o 'yolo+ml'")
            elif modo == "yolo+ml" and not estado["ml_disponible"]:
                errores.append("el clasificador ML no está entrenado (python clasificador_ml.py)")
            else:
                p["modo"] = modo
        return dict(p), errores


def capturar_roi():
    """Guarda la última ROI de la mano en capturas/roi_<fecha>.jpg y dice de qué seña era.
    La usan el botón de la web y la herramienta capturar_roi del agente."""
    if jpeg_roi is None:
        return {"error": "Todavía no he visto ninguna mano: pon la mano frente a la cámara y vuelve a pedirlo."}
    datos, info = jpeg_roi, dict(info_roi)
    CAPTURAS.mkdir(exist_ok=True)
    nombre = f"roi_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
    (CAPTURAS / nombre).write_bytes(datos)
    y, ml = info["yolo"], info["ml"]
    hace = time.time() - info["t"]
    mensaje = f"Guardé la ROI de tu mano ({nombre}). YOLO vio «{y['clase']}» con {y['conf']:.2f}"
    if ml:
        mensaje += f" y el ML dijo «{ml['clase']}» ({ml['prob']:.2f})"
    mensaje += "." if hace < 2 else f". Es de hace {hace:.0f} s (la última vez que vi una mano)."
    return {"archivo": f"capturas/{nombre}", "url": f"/capturas/{nombre}", "sena_yolo": y["clase"],
            "conf_yolo": y["conf"], "ml": ml and {"clase": ml["clase"], "prob": ml["prob"]},
            "hace_segundos": round(hace, 1), "mensaje": mensaje}


def foto_estado():
    with candado:
        return json.loads(json.dumps(estado))      # copia profunda segura para mandar como JSON


# ------------------------------------------------------------------ Flask
def crear_app(agente):
    from flask import Flask, Response, abort, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def inicio():
        return send_from_directory(BASE / "web", "index.html")

    @app.get("/video")
    def video():
        """Stream MJPEG: el navegador recibe JPEG uno tras otro en la misma respuesta
        (multipart/x-mixed-replace) y los va reemplazando: parece video, sin librerías."""
        def generar():
            while activo:
                if jpeg_video:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_video + b"\r\n"
                time.sleep(1 / 15)
        return Response(generar(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/roi.jpg")
    def roi_jpg():
        img = jpeg_roi or cv2.imencode(".jpg", cuadro_mensaje("Aun no se ha visto una mano"))[1].tobytes()
        return Response(img, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/roi")
    def api_roi():
        """Botón «Guardar ROI»: guarda el recorte y el agente lo muestra en el chat."""
        r = capturar_roi()
        agente.mostrar_roi(r)
        return jsonify(ok="error" not in r, **r)

    @app.get("/capturas/<archivo>")
    def capturas(archivo):
        if ".." in archivo or "/" in archivo or "\\" in archivo:
            abort(404)
        return send_from_directory(CAPTURAS, archivo)

    @app.get("/etapas.jpg")
    def etapas():
        img = jpeg_etapas or cv2.imencode(".jpg", cuadro_mensaje("Aun no se ha visto una mano"))[1].tobytes()
        return Response(img, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/estado")
    def api_estado():
        datos = foto_estado()
        datos["decision"] = registros[-1] if registros else None
        datos["eventos"] = list(eventos)[-20:][::-1]
        datos["chat"] = agente.chat_reciente()
        return jsonify(datos)

    @app.post("/api/comando")
    def api_comando():
        letra = str((request.get_json(silent=True) or {}).get("letra", "")).upper()
        if letra not in ("L", "V", "S", "T", "?"):
            return jsonify(ok=False, error="letra debe ser L, V, S, T o ?"), 400
        antes, despues = enviar(letra, "manual")
        if letra != "?":
            agente.avisar_hecho(letra, "manual", antes, despues, None, None)
        return jsonify(ok=True, actuadores=despues)

    @app.post("/api/parametros")
    def api_parametros():
        try:
            p, errores = ajustar_parametros(request.get_json(silent=True) or {})
        except (TypeError, ValueError):
            return jsonify(ok=False, error="valores no numéricos"), 400
        return jsonify(ok=not errores, parametros=p, errores=errores)

    @app.get("/api/metricas")
    def api_metricas():
        return jsonify(metricas())

    @app.get("/runs/<archivo>")
    def runs(archivo):
        if ".." in archivo or "/" in archivo or "\\" in archivo:
            abort(404)
        return send_from_directory(RUNS, archivo)       # send_from_directory no deja salir de la carpeta

    @app.post("/api/agente")
    def api_agente():
        mensaje = str((request.get_json(silent=True) or {}).get("mensaje", "")).strip()
        if not mensaje:
            return jsonify(ok=False, error="mensaje vacío"), 400
        return jsonify(ok=True, **agente.preguntar(mensaje))

    @app.post("/api/agente/reiniciar")
    def api_agente_reiniciar():
        agente.reiniciar()
        return jsonify(ok=True)

    return app


# ------------------------------------------------------------------ arranque
def main():
    global pico, activo
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", default="0", help="0 integrada, 1 externa... o ruta de un video")
    ap.add_argument("--puerto", default=None, help="Puerto de la Pico (COM5). Por defecto la busca")
    ap.add_argument("--modelo", default=str(RUNS / "weights" / "best.pt"))
    ap.add_argument("--simular", action="store_true", help="no usar la Pico: simulador que imita main.cpp")
    ap.add_argument("--puerto-web", type=int, default=5000)
    args = ap.parse_args()

    if not Path(args.modelo).exists():
        raise SystemExit(f"No existe el modelo YOLO '{args.modelo}'. Entrénalo con: python entrenar_senas.py")

    estado["ml_disponible"] = clasificador_ml.cargar()
    if not estado["ml_disponible"]:
        estado["parametros"]["modo"] = "yolo"
        estado["aviso"] = "Clasificador ML no entrenado: modo solo YOLO (corre python clasificador_ml.py)"
        print("[AVISO]", estado["aviso"])

    # Pico real si se puede; si no, simulador (la demo nunca se cae por la Pico)
    if args.simular:
        pasar_a_simulado("--simular")
    else:
        try:
            pico = abrir_pico(args.puerto)
        except SystemExit as e:
            print(f"[AVISO] {e} ¿Está abierto el monitor serial de PlatformIO o control_senas.py?")
            pico = None
        if pico is None:
            pasar_a_simulado("Pico no encontrada o puerto ocupado")
        else:
            estado["simulado"] = False
            estado["conexion"] = f"Pico conectada ({pico.port})"

    agente = agente_mod.Agente(agente_mod.Sistema(
        estado=foto_estado, enviar=enviar, registros=lambda n: list(registros)[-n:],
        eventos=lambda n: list(eventos)[-n:], metricas=metricas, ajustar=ajustar_parametros,
        capturar_roi=capturar_roi))

    threading.Thread(target=hilo_lector_serial, daemon=True).start()
    enviar("?", "inicio")                                      # sincroniza el estado inicial
    threading.Thread(target=hilo_vision, args=(args.camara, args.modelo, agente), daemon=True).start()

    print(f"\nAbre  http://localhost:{args.puerto_web}   (Ctrl+C para salir)\n")
    try:
        # debug=False SIEMPRE: el recargador de debug arranca el proceso dos veces
        # y abriría la cámara y el puerto COM dos veces.
        crear_app(agente).run(host="0.0.0.0", port=args.puerto_web, threaded=True, debug=False)
    finally:
        activo = False
        time.sleep(0.3)
        try:
            pico.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
