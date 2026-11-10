"""
PRUEBAS DEL SISTEMA (rúbrica: calidad del resultado y pruebas).

Sobre las fotos de validación (dataset_senas/images/val, que YOLO y el ML NO vieron al entrenar):
  - compara YOLO solo, ML solo (con la caja real y con la caja de YOLO) y la fusión YOLO+ML,
  - cuenta falsos positivos en las fotos NEGATIVAS (sin seña),
  - repite todo con variaciones razonables: más oscura/clara (±40 %), desenfoque, ruido, giro ±15° y espejo,
  - sugiere ML_MIN, NITIDEZ_MIN y BRILLO_MIN,
  - guarda modelos/pruebas.json (lo muestra la web) y crea demo_val.mp4 (dato de ejemplo sin cámara).

Uso:  python evaluar_sistema.py
Las métricas son POR CUADRO; en vivo además se exige la misma seña en FRAMES_CONFIRMAR cuadros seguidos,
lo que reduce todavía más los disparos falsos.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import clasificador_ml
import procesamiento as proc
import servidor_web as sw
from control_senas import CONF_MIN

BASE = Path(__file__).resolve().parent
VAL = BASE / "dataset_senas"


def rotar(frame, caja, grados):
    """Gira la imagen alrededor del centro y devuelve también la caja girada (su rectángulo envolvente)."""
    H, W = frame.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), grados, 1.0)
    girada = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    if caja is None:
        return girada, None
    x1, y1, x2, y2 = caja
    esquinas = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]]) @ M.T
    return girada, (esquinas[:, 0].min(), esquinas[:, 1].min(), esquinas[:, 0].max(), esquinas[:, 1].max())


def ruido(frame):
    rng = np.random.default_rng(0)
    return np.clip(frame.astype(np.int16) + rng.normal(0, 20, frame.shape), 0, 255).astype(np.uint8)


# nombre -> función(frame, caja) -> (frame_modificado, caja_modificada)
VARIACIONES = {
    "original": lambda f, c: (f, c),
    "oscura -40%": lambda f, c: (cv2.convertScaleAbs(f, alpha=0.6), c),
    "clara +40%": lambda f, c: (cv2.convertScaleAbs(f, alpha=1.4), c),
    "desenfoque": lambda f, c: (cv2.GaussianBlur(f, (0, 0), 3), c),              # moderado: debe seguir funcionando
    "desenfoque fuerte": lambda f, c: (cv2.GaussianBlur(f, (0, 0), 9), c),       # inservible: el filtro de calidad debe frenarlo
    "ruido": lambda f, c: (ruido(f), c),
    "giro +15°": lambda f, c: rotar(f, c, 15),
    "giro -15°": lambda f, c: rotar(f, c, -15),
    "espejo": lambda f, c: (cv2.flip(f, 1), None if c is None else (f.shape[1] - c[2], c[1], f.shape[1] - c[0], c[3])),
}


def leer_val():
    """[(frame, clase | None, caja | None)] — clase None = foto negativa (sin seña)."""
    datos = []
    for img in sorted((VAL / "images" / "val").glob("*.jpg")):
        frame = cv2.imread(str(img))
        lab = VAL / "labels" / "val" / f"{img.stem}.txt"
        lineas = [l.split() for l in lab.read_text().splitlines() if l.strip()] if lab.exists() else []
        if not lineas:
            datos.append((frame, None, None))
            continue
        H, W = frame.shape[:2]
        c, cx, cy, w, h = int(lineas[0][0]), *map(float, lineas[0][1:5])
        datos.append((frame, clasificador_ml.CLASES[c], ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)))
    return datos


def evaluar_variacion(nombre, datos, yolo, params, dispositivo):
    r = {k: 0 for k in ("pos", "neg", "yolo_ok", "yolo_error", "yolo_fp", "ml_real_ok", "ml_yolo_ok", "ml_yolo_n",
                        "fusion_ok", "fusion_error", "fusion_fp", "fusion_rechaza_correcto", "rechazo_calidad")}
    probs_ok, probs_mal, nitideces = [], [], []
    for frame, clase, caja in datos:
        f, c = VARIACIONES[nombre](frame, caja)
        r["neg" if clase is None else "pos"] += 1

        # YOLO solo (sin filtro de calidad ni ML): la caja más segura con conf >= CONF_MIN
        res = yolo.predict(f, conf=params["conf_min"], verbose=False, device=dispositivo)[0]
        y_clase = res.names[int(res.boxes.cls[int(res.boxes.conf.argmax())])] if len(res.boxes) else None
        if clase is None:
            r["yolo_fp"] += y_clase is not None
        else:
            r["yolo_ok"] += y_clase == clase
            r["yolo_error"] += y_clase is not None and y_clase != clase

        # ML solo con la caja REAL (etiquetada a mano)
        if clase is not None:
            roi = proc.recortar_roi(f, c)
            vec, _ = proc.procesar_roi(roi) if roi is not None else (None, None)
            if vec is not None:
                r["ml_real_ok"] += clasificador_ml.predecir(vec)[0] == clase

        # Sistema completo (calidad -> YOLO -> procesamiento -> ML -> decisión), igual que en vivo
        reg, _ = sw.analizar_cuadro(f, yolo, params, dispositivo)
        nitideces.append(reg["calidad"]["nitidez"])
        r["rechazo_calidad"] += reg["etapa_falla"] == "calidad"
        if reg["ml"] and clase is not None:                     # ML sobre la caja de YOLO
            r["ml_yolo_n"] += 1
            r["ml_yolo_ok"] += reg["ml"]["clase"] == clase
            if reg["yolo"]["clase"] == clase:
                probs_ok.append(reg["ml"]["prob_yolo"])
            else:
                probs_mal.append(reg["ml"]["prob_yolo"])
        if clase is None:
            r["fusion_fp"] += reg["valido"]
        else:
            r["fusion_ok"] += reg["valido"] and reg["sena"] == clase
            r["fusion_error"] += reg["valido"] and reg["sena"] != clase
            r["fusion_rechaza_correcto"] += (not reg["valido"] and reg["yolo"] is not None
                                             and reg["yolo"]["clase"] == clase and reg["yolo"]["conf"] >= params["conf_min"])
    p = max(r["pos"], 1)
    fila = {
        "variacion": nombre, "positivas": r["pos"], "negativas": r["neg"],
        "yolo_exactitud": round(r["yolo_ok"] / p, 3), "yolo_errores": r["yolo_error"], "yolo_fp_negativas": r["yolo_fp"],
        "ml_caja_real": round(r["ml_real_ok"] / p, 3),
        "ml_caja_yolo": round(r["ml_yolo_ok"] / max(r["ml_yolo_n"], 1), 3),
        "fusion_exactitud": round(r["fusion_ok"] / p, 3), "fusion_errores": r["fusion_error"],
        "fusion_fp_negativas": r["fusion_fp"], "fusion_rechazos": r["fusion_rechaza_correcto"],
        "rechazo_calidad": r["rechazo_calidad"],
    }
    return fila, probs_ok, probs_mal, nitideces


def crear_video_demo(ruta, cuadros_por_foto=3, fps=10):
    """demo_val.mp4: las fotos de validación en orden (bloques de la misma seña seguidos), 1280x720.
    Cada foto dura 3 cuadros a 10 fps -> un bloque de 8 fotos son ~2.4 s de la misma seña."""
    salida = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    for img in sorted((VAL / "images" / "val").glob("*.jpg")):
        f = cv2.imread(str(img))
        s = min(1280 / f.shape[1], 720 / f.shape[0])
        r = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)))
        lienzo = np.full((720, 1280, 3), 114, np.uint8)        # bandas grises (negro bajaría el brillo medio)
        y0, x0 = (720 - r.shape[0]) // 2, (1280 - r.shape[1]) // 2
        lienzo[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        for _ in range(cuadros_por_foto):
            salida.write(lienzo)
    salida.release()


def main():
    import torch
    from ultralytics import YOLO

    if not clasificador_ml.cargar():
        raise SystemExit("Primero entrena el clasificador:  python clasificador_ml.py")
    sw.estado["ml_disponible"] = True
    dispositivo = 0 if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(sw.RUNS / "weights" / "best.pt"))
    params = {"conf_min": CONF_MIN, "frames_confirmar": 5, "ml_min": sw.ML_MIN, "modo": "yolo+ml"}
    datos = leer_val()
    print(f"Validación: {sum(c is not None for _, c, _ in datos)} fotos con seña, "
          f"{sum(c is None for _, c, _ in datos)} negativas | CONF_MIN {CONF_MIN} | ML_MIN {sw.ML_MIN} | "
          f"{'GPU' if dispositivo == 0 else 'CPU'}\n")

    filas, todos_ok, todos_mal, nit = [], [], [], {}
    for nombre in VARIACIONES:
        fila, ok, mal, nits = evaluar_variacion(nombre, datos, yolo, params, dispositivo)
        filas.append(fila)
        todos_ok += ok
        todos_mal += mal
        nit[nombre] = nits
        print(f"  {nombre} listo")

    cols = [("variacion", "Variación", 17), ("yolo_exactitud", "YOLO", 6), ("yolo_errores", "Err", 4),
            ("yolo_fp_negativas", "FP neg", 7), ("ml_caja_real", "ML real", 8), ("ml_caja_yolo", "ML yolo", 8),
            ("fusion_exactitud", "Fusión", 7), ("fusion_errores", "Err", 4), ("fusion_fp_negativas", "FP neg", 7),
            ("fusion_rechazos", "Rechaza", 8), ("rechazo_calidad", "Calidad", 8)]
    print("\n" + " ".join(f"{t:>{w}s}" for _, t, w in cols))
    for f in filas:
        print(" ".join(f"{str(f[k]):>{w}s}" for k, _, w in cols))
    print("\nYOLO/ML/Fusión = exactitud sobre fotos con seña. Err = acciones EQUIVOCADAS (otra seña).")
    print("FP neg = fotos SIN seña donde se habría accionado. Rechaza = YOLO acertaba pero la fusión no aceptó.")
    print("Calidad = cuadros descartados por borrosos u oscuros.")

    # ---- umbrales sugeridos ----
    ok, mal = np.array(todos_ok), np.array(todos_mal)
    p5 = float(np.percentile(ok, 5)) if len(ok) else 0.5
    ml_min = round(min(0.5, max(0.3, p5 - 0.05)) * 20) / 20         # múltiplo de 0.05 entre 0.30 y 0.50
    orig = np.array(nit["original"])
    sugeridos = {
        "ML_MIN": ml_min,
        # a medio camino (en escala logarítmica) entre el desenfoque moderado y el fuerte
        "NITIDEZ_MIN": int(round(np.sqrt(np.median(nit["desenfoque"]) * np.median(nit["desenfoque fuerte"])))),
        "BRILLO_MIN": proc.BRILLO_MIN,
        "explicacion": {
            "ML_MIN": f"el 95 % de los cuadros donde YOLO acierta tienen probabilidad ML ≥ {p5:.2f}; "
                      f"cuando YOLO se equivoca la mediana es {np.median(mal) if len(mal) else float('nan'):.2f}",
            "NITIDEZ_MIN": f"val original: 1 % más borroso {np.percentile(orig, 1):.0f}; mediana con desenfoque "
                           f"moderado {np.median(nit['desenfoque']):.0f} y fuerte {np.median(nit['desenfoque fuerte']):.0f}",
            "BRILLO_MIN": "las fotos oscurecidas 40 % siguen teniendo brillo > 80, así que 50 solo rechaza cuadros casi negros",
        },
    }
    print(f"\nUmbrales sugeridos: ML_MIN={sugeridos['ML_MIN']}  NITIDEZ_MIN={sugeridos['NITIDEZ_MIN']}  "
          f"BRILLO_MIN={sugeridos['BRILLO_MIN']}")
    for k, v in sugeridos["explicacion"].items():
        print(f"  {k}: {v}")
    print(f"  (en uso ahora: ML_MIN={sw.ML_MIN}, NITIDEZ_MIN={proc.NITIDEZ_MIN}, BRILLO_MIN={proc.BRILLO_MIN})")

    o = filas[0]
    resumen = {"yolo_exactitud": o["yolo_exactitud"], "fusion_exactitud": o["fusion_exactitud"],
               "yolo_fp_negativas": o["yolo_fp_negativas"], "fusion_fp_negativas": o["fusion_fp_negativas"],
               "negativas": o["negativas"], "positivas": o["positivas"],
               "peor_variacion": min(filas, key=lambda f: f["fusion_exactitud"])["variacion"]}
    (BASE / "modelos").mkdir(exist_ok=True)
    (BASE / "modelos" / "pruebas.json").write_text(json.dumps({
        "parametros": params, "variaciones": filas, "sugeridos": sugeridos, "resumen": resumen,
        "nota": "Métricas por cuadro sobre las fotos de validación; en vivo además se exige la seña en varios cuadros seguidos."},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nGuardado: modelos/pruebas.json")

    crear_video_demo(BASE / "demo_val.mp4")
    print("Creado: demo_val.mp4  (python servidor_web.py --camara demo_val.mp4 --simular)")


if __name__ == "__main__":
    main()
