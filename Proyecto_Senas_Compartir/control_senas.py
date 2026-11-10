"""
Paso 4: SISTEMA COMPLETO en tiempo real.
cámara -> modelo de señas -> seña confirmada -> letra por USB a la Raspberry Pi Pico -> cambia el actuador

Uso:
  python control_senas.py                       # cámara 0 (integrada), busca la Pico sola
  python control_senas.py --camara 1            # cámara USB externa
  python control_senas.py --puerto COM5         # si no encuentra la Pico sola
  python control_senas.py --camara prueba.mp4   # probar con un video

Cada seña CAMBIA el estado de su actuador (apagado <-> encendido).
Para repetir la MISMA seña, baja la mano un momento entre una y otra; una seña distinta se acepta enseguida.
Tecla q = salir.  Cierra el monitor serial de PlatformIO antes de correr esto (ocupa el puerto).
"""
import argparse
import threading
import time

import cv2
import serial
from serial.tools import list_ports
from ultralytics import YOLO

# Nombre de la clase (el mismo de X-AnyLabeling) -> letra que entiende la Pico
COMANDOS = {"led": "L", "ventilador": "V", "servo": "S", "todo": "T"}
CONF_MIN = 0.6          # confianza mínima para aceptar una seña
FRAMES_CONFIRMAR = 5    # la misma seña en 5 cuadros seguidos = confirmada (evita disparos por error)
TIEMPO_LIBRE = 0.8      # segundos sin seña para poder repetir la misma


class Camara:
    """Lee la cámara en un hilo aparte y guarda solo el último cuadro:
    aunque el modelo sea lento, siempre se procesa la imagen actual (sin retraso)."""

    def __init__(self, fuente):
        self.es_video = not fuente.isdigit()
        if self.es_video:
            self.cap = cv2.VideoCapture(fuente)
        else:
            self.cap = cv2.VideoCapture(int(fuente), cv2.CAP_DSHOW)   # DSHOW: abre rápido en Windows
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(int(fuente))
        if not self.cap.isOpened():
            raise SystemExit(f"No se pudo abrir la cámara/video '{fuente}'")
        self.ok, self.frame, self.activa = False, None, True
        if not self.es_video:
            threading.Thread(target=self._bucle, daemon=True).start()

    def _bucle(self):
        while self.activa:
            self.ok, self.frame = self.cap.read()

    def leer(self):
        if self.es_video:                       # en un video se procesan todos los cuadros en orden
            return self.cap.read()
        frame = self.frame
        return self.ok and frame is not None, None if frame is None else frame.copy()

    def cerrar(self):
        self.activa = False
        time.sleep(0.1)
        self.cap.release()


def abrir_pico(puerto):
    if not puerto:   # busca la Pico por su fabricante USB (0x2E8A = Raspberry Pi)
        puerto = next((p.device for p in list_ports.comports() if p.vid == 0x2E8A), None)
    if not puerto:
        print("[AVISO] No se encontró la Pico: el programa corre igual, pero solo imprime las órdenes.")
        return None
    try:
        pico = serial.Serial(puerto, 115200, timeout=0)
    except serial.SerialException as e:
        raise SystemExit(f"No se pudo abrir {puerto} ({e}). ¿Está abierto el monitor serial de PlatformIO?")
    print(f"Pico conectada en {puerto}")
    time.sleep(0.5)
    pico.write(b"?")          # pide el estado actual de los actuadores
    return pico


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", default="0", help="0 integrada, 1 externa... o ruta de un video")
    ap.add_argument("--puerto", default=None, help="Puerto de la Pico (COM5, /dev/ttyACM0). Por defecto la busca")
    ap.add_argument("--modelo", default="runs/senas/weights/best.pt")
    args = ap.parse_args()

    modelo = YOLO(args.modelo)
    faltan = set(COMANDOS) - set(modelo.names.values())
    if faltan:
        print(f"[AVISO] El modelo no tiene estas clases: {faltan}. Revisa los nombres en COMANDOS.")
    pico = abrir_pico(args.puerto)
    cam = Camara(args.camara)

    candidata, cuenta = None, 0       # seña que se está viendo y cuántos cuadros seguidos lleva
    bloqueada, t_visto = None, 0.0    # última seña enviada: no se repite hasta bajar la mano
    estado, buffer = "", ""
    fps, t_prev = 0.0, time.time()
    ventana = "Senas (q = salir)"
    print("Cámara abierta. Pulsa q en la ventana para salir.")

    try:
        while True:
            ok, frame = cam.leer()
            if not ok:
                if cam.es_video:
                    break
                time.sleep(0.01)
                continue

            # --- detectar la seña más segura del cuadro ---
            res = modelo.predict(frame, conf=CONF_MIN, verbose=False)[0]
            sena = None
            if len(res.boxes):
                b = res.boxes[int(res.boxes.conf.argmax())]
                sena = res.names[int(b.cls[0])]
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 3)
                cv2.putText(frame, f"{sena} {float(b.conf[0]):.2f}", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

            # --- confirmar la seña y mandar la orden ---
            ahora = time.time()
            cuenta = cuenta + 1 if sena == candidata else 1
            candidata = sena
            if sena:
                t_visto = ahora
            elif ahora - t_visto > TIEMPO_LIBRE:
                bloqueada = None                      # mano abajo: ya se puede repetir la misma seña
            if sena in COMANDOS and cuenta >= FRAMES_CONFIRMAR and sena != bloqueada:
                bloqueada = sena
                print(f"Seña confirmada: {sena} -> envío '{COMANDOS[sena]}'")
                if pico:
                    pico.write(COMANDOS[sena].encode())

            # --- leer lo que responde la Pico (estado de los actuadores) ---
            if pico and pico.in_waiting:
                buffer += pico.read(pico.in_waiting).decode(errors="ignore")
                while "\n" in buffer:
                    linea, buffer = buffer.split("\n", 1)
                    if linea.strip():
                        estado = linea.strip()
                        print("Pico ->", estado)

            # --- pantalla ---
            fps = 0.9 * fps + 0.1 / max(1e-6, ahora - t_prev)
            t_prev = ahora
            cv2.putText(frame, f"{fps:4.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, estado or ("Sin Pico" if not pico else "..."), (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            cv2.imshow(ventana, frame)
            if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.cerrar()
        if pico:
            pico.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
