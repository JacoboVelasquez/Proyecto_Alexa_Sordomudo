"""
TEMA 2 — MACHINE LEARNING CLÁSICO.

Entrena un clasificador de scikit-learn con las características de procesamiento.py
(forma del contorno + momentos de Hu + HOG). En el sistema en vivo este modelo VERIFICA
la seña que dice YOLO: si los dos no están de acuerdo, no se acciona nada.

Uso:
  python clasificador_ml.py        # entrena y guarda modelos/clasificador_ml.joblib + metricas_ml.json
"""
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

import procesamiento as proc

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset_senas"
CARPETA_MODELOS = BASE / "modelos"
RUTA_MODELO = CARPETA_MODELOS / "clasificador_ml.joblib"
RUTA_METRICAS = CARPETA_MODELOS / "metricas_ml.json"
CLASES = ["led", "ventilador", "servo", "todo"]   # mismo orden que preparar_dataset.py (id YOLO 0-3)


def leer_cajas(split):
    """Recorre dataset_senas/labels/<split>: devuelve [(ruta_imagen, clase, (x1, y1, x2, y2)), ...].
    Las etiquetas YOLO vienen normalizadas (cx, cy, w, h en 0-1); aquí se pasan a píxeles."""
    salida = []
    for lab in sorted((DATASET / "labels" / split).glob("*.txt")):
        img = DATASET / "images" / split / f"{lab.stem}.jpg"
        lineas = [l.split() for l in lab.read_text().splitlines() if l.strip()]
        if not lineas or not img.exists():
            continue                                   # negativas (sin mano) no sirven para entrenar el ML
        H, W = cv2.imread(str(img)).shape[:2]
        for c, cx, cy, w, h in ([int(l[0])] + list(map(float, l[1:5])) for l in lineas):
            caja = ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)
            salida.append((img, CLASES[c], caja))
    return salida


def armar_datos(split, aumentar=False):
    """Para cada mano etiquetada: recorte -> características. Con aumentar=True agrega
    el espejo horizontal (igual que fliplr de YOLO: mano izquierda/derecha) y versiones
    un 25 % más oscura y más clara (cambios de iluminación)."""
    X, y, sin_contorno = [], [], 0
    cache = {}
    for img, clase, caja in leer_cajas(split):
        frame = cache.setdefault(img, cv2.imread(str(img)))
        roi = proc.recortar_roi(frame, caja)
        variantes = [roi]
        if aumentar:
            variantes += [cv2.flip(roi, 1),
                          cv2.convertScaleAbs(roi, alpha=0.75),
                          cv2.convertScaleAbs(roi, alpha=1.25)]
        for v in variantes:
            vector, _ = proc.procesar_roi(v)
            if vector is None:
                sin_contorno += 1
                continue
            X.append(vector)
            y.append(clase)
    return np.array(X), np.array(y), sin_contorno


def candidatos():
    """Los modelos que se comparan. Todos van en un Pipeline con StandardScaler, porque las
    características tienen escalas muy distintas (área ~0.3, Hu ~3, HOG ~0.1)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return {
        # La SVM no da probabilidades por sí sola: CalibratedClassifierCV las aprende con validación cruzada
        "SVM (RBF)": Pipeline([("escala", StandardScaler()),
                               ("modelo", CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"),
                                                                 ensemble=False))]),
        "Random Forest": Pipeline([("escala", StandardScaler()),
                                   ("modelo", RandomForestClassifier(n_estimators=300, random_state=0))]),
        "KNN (k=5)": Pipeline([("escala", StandardScaler()),
                               ("modelo", KNeighborsClassifier(n_neighbors=5, weights="distance"))]),
    }


def entrenar():
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    print("Extrayendo características (mismo split train/val que YOLO)...")
    Xtr, ytr, sc_tr = armar_datos("train", aumentar=True)
    Xva, yva, sc_va = armar_datos("val")
    print(f"  train: {len(ytr)} ejemplos (con aumentos) | val: {len(yva)} | sin contorno: {sc_tr} / {sc_va}")
    print(f"  {Xtr.shape[1]} características por mano\n")

    resultados, mejor, mejor_nombre, modelos = {}, None, None, {}
    print(f"{'Modelo':16s} {'Exactitud train':>16s} {'Exactitud val':>14s}")
    for nombre, pipe in candidatos().items():
        pipe.fit(Xtr, ytr)
        modelos[nombre] = pipe
        a_tr = accuracy_score(ytr, pipe.predict(Xtr))
        a_va = accuracy_score(yva, pipe.predict(Xva))
        resultados[nombre] = {"train": round(a_tr, 4), "val": round(a_va, 4)}
        print(f"{nombre:16s} {a_tr:16.3f} {a_va:14.3f}")
        if mejor is None or a_va > resultados[mejor_nombre]["val"]:
            mejor, mejor_nombre = pipe, nombre

    pred = mejor.predict(Xva)
    matriz = confusion_matrix(yva, pred, labels=CLASES)
    print(f"\nElegido: {mejor_nombre} (mejor exactitud en val)\n")
    print(classification_report(yva, pred, labels=CLASES, digits=3, zero_division=0))
    print("Matriz de confusión (filas = real, columnas = predicha):")
    print("            " + " ".join(f"{c[:6]:>7s}" for c in CLASES))
    for c, fila in zip(CLASES, matriz):
        print(f"  {c:10s}" + " ".join(f"{v:7d}" for v in fila))

    # El Random Forest dice qué características usó más (aunque no sea el elegido, sirve para explicar)
    imp = modelos["Random Forest"].named_steps["modelo"].feature_importances_
    nombres = proc.NOMBRES_CARACTERISTICAS
    top = np.argsort(imp)[::-1][:10]
    importancias = {
        "top10": [{"nombre": nombres[i], "importancia": round(float(imp[i]), 4)} for i in top],
        "por_grupo": {"forma": round(float(imp[:6].sum()), 3), "hu": round(float(imp[6:13].sum()), 3),
                      "hog": round(float(imp[13:].sum()), 3)},
    }
    print("\nImportancia por grupo (Random Forest):", importancias["por_grupo"])

    CARPETA_MODELOS.mkdir(exist_ok=True)
    joblib.dump({"modelo": mejor, "nombre": mejor_nombre, "clases": list(mejor.classes_)}, RUTA_MODELO)
    metricas = {
        "modelo_elegido": mejor_nombre,
        "exactitud_val": resultados[mejor_nombre]["val"],
        "comparacion": resultados,
        "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "n_caracteristicas": int(Xtr.shape[1]),
        "clases": CLASES,
        "matriz_confusion": matriz.tolist(),
        "reporte": classification_report(yva, pred, labels=CLASES, output_dict=True, zero_division=0),
        "importancias_random_forest": importancias,
        "sin_contorno": {"train": sc_tr, "val": sc_va},
    }
    RUTA_METRICAS.write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado: {RUTA_MODELO.relative_to(BASE)} y {RUTA_METRICAS.relative_to(BASE)}")


# ----------------------------- uso en vivo -----------------------------
_modelo = None


def cargar():
    """Carga el clasificador guardado. Devuelve True si existe, False si hay que entrenarlo."""
    global _modelo
    if not RUTA_MODELO.exists():
        return False
    _modelo = joblib.load(RUTA_MODELO)
    return True


def predecir(vector):
    """vector -> (clase_predicha, {clase: probabilidad})."""
    if _modelo is None and not cargar():
        raise RuntimeError("Falta modelos/clasificador_ml.joblib: corre  python clasificador_ml.py")
    p = _modelo["modelo"].predict_proba(vector.reshape(1, -1))[0]
    probs = {c: float(v) for c, v in zip(_modelo["clases"], p)}
    return max(probs, key=probs.get), probs


if __name__ == "__main__":
    entrenar()
