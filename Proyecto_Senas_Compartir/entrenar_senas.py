"""Paso 3: entrena el detector de señas. Corre primero preparar_dataset.py
Resultado: runs/senas/weights/best.pt
"""
from pathlib import Path

import torch
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

if __name__ == "__main__":          # necesario en Windows (workers del dataloader)
    gpu = torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0) if gpu else "NO (va a ser lento)")
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(BASE / "dataset_senas" / "data.yaml"),
        epochs=100,
        imgsz=640,
        batch=16 if gpu else 8,     # si sale "out of memory", baja a 8
        device=0 if gpu else "cpu",
        patience=25,
        project=str(BASE / "runs"),
        name="senas",
        workers=2,
        exist_ok=True,
        fliplr=0.5,                 # espejo = mano izquierda/derecha. Si una seña depende de la
                                    # dirección (p. ej. apuntar a la izquierda vs. derecha), pon 0.0
        # amp=False,                # descomentar si en la GTX 1650 las pérdidas salen "nan"
    )
    m = model.val()
    print(f"\nVal -> P {m.box.mp:.3f} | R {m.box.mr:.3f} | mAP50 {m.box.map50:.3f} | mAP50-95 {m.box.map:.3f}")
    print("Modelo: runs/senas/weights/best.pt")
