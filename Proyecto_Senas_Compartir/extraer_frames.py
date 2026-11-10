"""
Paso 1: guarda 1 de cada N cuadros de un video en images/ (la misma carpeta para todos los videos).

Uso:
    python extraer_frames.py video_led.mp4
    python extraer_frames.py video_todo.mp4 --cada 5
Usa nombres de video SIN tildes ni ñ (OpenCV en Windows falla con esas rutas).
"""
import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--cada", type=int, default=10, help="guardar 1 cuadro de cada N")
    ap.add_argument("--salida", default="images")
    a = ap.parse_args()

    src, salida = Path(a.video), Path(a.salida)
    salida.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {src}")

    i = guardadas = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % a.cada == 0:
            cv2.imwrite(str(salida / f"{src.stem}_{guardadas:04d}.jpg"), frame)
            guardadas += 1
        i += 1
    cap.release()
    print(f"Guardadas {guardadas} imágenes en {salida.resolve()}")


if __name__ == "__main__":
    main()
