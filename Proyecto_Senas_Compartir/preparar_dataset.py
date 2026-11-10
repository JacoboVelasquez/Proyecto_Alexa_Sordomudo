"""
Paso 2: convierte los JSON de X-AnyLabeling (images/) a formato YOLO y arma
dataset_senas/ con train/val + data.yaml.

Uso:  python preparar_dataset.py
"""
import json
import random
import shutil
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "images"
DST = BASE / "dataset_senas"
# Etiquetas EXACTAS que se usan en X-AnyLabeling. El orden define el id de clase (0, 1, 2, 3).
# Si cambias los nombres aquí, cámbialos también en COMANDOS de control_senas.py
CLASES = ["led", "ventilador", "servo", "todo"]
VAL_FRAC = 0.2
BLOQUE = 8           # cuadros seguidos que van juntos a train o a val (evita un val "contaminado")


def json_a_yolo(js):
    d = json.loads(js.read_text(encoding="utf-8"))
    W, H = d["imageWidth"], d["imageHeight"]
    lineas = []
    for s in d.get("shapes", []):
        etiqueta = str(s.get("label", "")).strip().lower()
        if etiqueta not in CLASES:
            print(f"  [!] etiqueta desconocida '{s.get('label')}' en {js.name}")
            continue
        xs = [p[0] for p in s["points"]]
        ys = [p[1] for p in s["points"]]
        x1, x2 = max(0, min(xs)), min(W, max(xs))
        y1, y2 = max(0, min(ys)), min(H, max(ys))
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        w, h = (x2 - x1) / W, (y2 - y1) / H
        lineas.append(f"{CLASES.index(etiqueta)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lineas


def main():
    imgs = sorted(p for p in SRC.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    ok, sin_json = [], 0
    for img in imgs:
        js = img.with_suffix(".json")
        if not js.exists():          # sin JSON = descartada a propósito (seña a medio hacer, borrosa...)
            sin_json += 1
            continue
        ok.append((img, json_a_yolo(js)))   # JSON sin cajas = negativo (mano sin seña / sin mano)

    # Split por bloques de cuadros consecutivos (cuadros vecinos son casi idénticos),
    # y POR SEÑA: cada seña (y los negativos) aporta ~20 % de sus bloques a val,
    # así ninguna se queda sin ejemplos de validación.
    por_sena = {}
    for img, lineas in ok:
        sena = CLASES[int(lineas[0].split()[0])] if lineas else "nada"
        por_sena.setdefault(sena, []).append((img, lineas))
    bloques, val_ids = {}, set()
    random.seed(42)
    for sena, items in por_sena.items():
        ids = []
        for i, item in enumerate(items):
            bid = (sena, i // BLOQUE)
            bloques.setdefault(bid, []).append(item)
            if bid not in ids:
                ids.append(bid)
        random.shuffle(ids)
        n_val = max(1, round(len(ids) * VAL_FRAC)) if len(ids) >= 2 else 0
        val_ids.update(ids[:n_val])

    if DST.exists():
        shutil.rmtree(DST)
    cuenta = {"train": 0, "val": 0}
    clases = {"train": Counter(), "val": Counter()}
    for b, items in bloques.items():
        split = "val" if b in val_ids else "train"
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lineas in items:
            shutil.copy2(img, DST / "images" / split / img.name)
            (DST / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lineas), encoding="utf-8")
            cuenta[split] += 1
            clases[split].update(CLASES[int(l.split()[0])] for l in lineas)

    nombres = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASES))
    (DST / "data.yaml").write_text(
        f"path: {DST.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{nombres}\n",
        encoding="utf-8",
    )

    print(f"Imágenes: {len(imgs)} | usadas {len(ok)} -> train {cuenta['train']} / val {cuenta['val']}")
    print(f"Sin JSON (descartadas): {sin_json}")
    print("Cajas por seña (train / val):")
    for c in CLASES:
        aviso = "   <-- ¡sin ejemplos en val!" if clases["val"][c] == 0 else ""
        print(f"  {c:12s} {clases['train'][c]:4d} / {clases['val'][c]:4d}{aviso}")
    print(f"\nListo: {DST / 'data.yaml'}")


if __name__ == "__main__":
    main()