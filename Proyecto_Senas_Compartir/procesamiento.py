"""
TEMA 1 — PROCESAMIENTO BÁSICO / INTERMEDIO DE IMÁGENES.

Toma la región de la mano que encontró YOLO y la procesa con técnicas clásicas:
  calidad del cuadro -> recorte (ROI) -> CLAHE -> YCrCb -> binarización Otsu -> morfología
  -> contorno más grande -> casco convexo y defectos -> vector de características (para el ML).

Todo usa solo OpenCV y NumPy (sin dependencias nuevas).
"""
import cv2
import numpy as np

# --- Umbrales de calidad (calibrados con evaluar_sistema.py sobre el dataset) ---
# Nitidez medida a 320 px de ancho: en el val original va de ~400 a ~1500; con un desenfoque
# moderado (que YOLO aún reconoce) baja a ~60 y con uno fuerte (imagen inservible) a < 10.
NITIDEZ_MIN = 20      # varianza del Laplaciano; menos = imagen borrosa (mano moviéndose, desenfocada)
BRILLO_MIN = 50       # media del canal V de HSV (0-255); menos = muy oscura

TAM_ROI = 128         # la mano se normaliza a 128x128 para que las características no dependan de la distancia
AREA_MIN = 0.08       # el contorno debe ocupar al menos 8 % del recorte, si no "no hay mano útil"
PROF_DEFECTO = 0.12   # un defecto de convexidad cuenta como "espacio entre dedos" si es más hondo que 12 % del lado

# HOG: imagen de 64x64 dividida en 4x4 celdas de 16 px, 9 orientaciones por celda -> 144 valores.
# (OpenCV 5 ya no trae cv2.HOGDescriptor, así que se implementa abajo en pocas líneas: hog_simple)
CELDA_HOG, BINS_HOG = 16, 9

NOMBRES_FORMA = ["area_rel", "aspecto", "extension", "solidez", "circularidad", "defectos"]
NOMBRES_HU = [f"hu{i + 1}" for i in range(7)]
NOMBRES_HOG = [f"hog{i}" for i in range((64 // CELDA_HOG) ** 2 * BINS_HOG)]
NOMBRES_CARACTERISTICAS = NOMBRES_FORMA + NOMBRES_HU + NOMBRES_HOG


def calidad_cuadro(frame):
    """Mide si el cuadro sirve antes de gastar tiempo en él.
    - Nitidez = varianza del Laplaciano: el Laplaciano resalta bordes; una imagen borrosa casi no tiene
      bordes, así que la varianza es baja. Se calcula a 320 px de ancho (INTER_AREA promedia los píxeles)
      para que no dependa de la resolución de la cámara ni del ruido fino del sensor.
    - Brillo = media del canal V (Value) de HSV, que es la intensidad de luz sin importar el color.
    Devuelve (nitidez, brillo, motivo); motivo es None si el cuadro está bien."""
    h, w = frame.shape[:2]
    pequeno = cv2.resize(frame, (320, max(1, int(320 * h / w))), interpolation=cv2.INTER_AREA)
    gris = cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)
    nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
    brillo = float(cv2.cvtColor(pequeno, cv2.COLOR_BGR2HSV)[..., 2].mean())
    motivo = None
    if brillo < BRILLO_MIN:
        motivo = f"poca luz (brillo {brillo:.0f} < {BRILLO_MIN})"
    elif nitidez < NITIDEZ_MIN:
        motivo = f"imagen borrosa (nitidez {nitidez:.0f} < {NITIDEZ_MIN})"
    return nitidez, brillo, motivo


def recortar_roi(frame, caja, margen=0.1):
    """Recorta la región de interés (ROI) de la mano: la caja (x1, y1, x2, y2) agrandada un 10 %
    por lado para no cortar las puntas de los dedos, y limitada a los bordes de la imagen."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = caja
    mx, my = (x2 - x1) * margen, (y2 - y1) * margen
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return frame[y1:y2, x1:x2]


def _cuadrar(roi):
    """Redimensiona a TAM_ROI x TAM_ROI conservando la proporción (rellena con negro)."""
    h, w = roi.shape[:2]
    s = TAM_ROI / max(h, w)
    r = cv2.resize(roi, (max(1, int(w * s)), max(1, int(h * s))))
    lienzo = np.zeros((TAM_ROI, TAM_ROI, 3), np.uint8)
    y0, x0 = (TAM_ROI - r.shape[0]) // 2, (TAM_ROI - r.shape[1]) // 2
    lienzo[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return lienzo


def segmentar_mano(roi):
    """Separa la piel del fondo dentro del recorte.
    1. Normaliza tamaño (128x128).
    2. CLAHE (ecualización de histograma adaptativa) sobre la luminancia Y: mejora el contraste
       local cuando hay sombras o la imagen está lavada, sin tocar el color.
    3. YCrCb: separa la luz (Y) del color (Cr, Cb). La piel tiene Cr alto casi sin importar si
       hay mucha o poca luz, por eso se segmenta en Cr y no en RGB.
    4. Otsu sobre Cr: escoge el umbral automáticamente (el que mejor separa los dos grupos del
       histograma). Se comparó con un rango fijo de piel (Cr 133-173, Cb 77-127) en el dataset y Otsu
       dio máscaras más completas, sobre todo con fotos sobreexpuestas, porque se adapta a cada imagen.
    5. Morfología: apertura (borra puntitos sueltos) + cierre (tapa huecos dentro de la mano).
    6. Se queda con el contorno más grande (la mano).
    Devuelve (roi_128, mascara, contorno) o (roi_128, mascara, None) si no hay mano útil."""
    r = _cuadrar(roi)
    ycc = cv2.cvtColor(r, cv2.COLOR_BGR2YCrCb)
    ycc[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(ycc[..., 0])
    r = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)

    _, mascara = cv2.threshold(ycc[..., 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k, iterations=2)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return r, mascara, None
    c = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(c) < AREA_MIN * TAM_ROI * TAM_ROI:
        return r, mascara, None
    return r, mascara, c


def hog_simple(gris):
    """HOG (histograma de orientaciones del gradiente), versión corta:
    1. Gradientes con Sobel en x e y -> magnitud y ángulo de cada píxel (cv2.cartToPolar).
    2. El ángulo se toma de 0 a 180° (un borde claro->oscuro y oscuro->claro cuentan igual).
    3. Por cada celda de 16x16 se arma un histograma de 9 orientaciones, pesado por la magnitud.
    4. Cada histograma se normaliza (L2) para que no dependa del contraste/luz.
    Resultado: dice hacia dónde apuntan los bordes en cada zona de la mano (dedos, nudillos)."""
    g = cv2.resize(gris, (64, 64)).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180
    bins = np.minimum((ang / (180 / BINS_HOG)).astype(int), BINS_HOG - 1)
    salida = []
    for y in range(0, 64, CELDA_HOG):
        for x in range(0, 64, CELDA_HOG):
            b = bins[y:y + CELDA_HOG, x:x + CELDA_HOG].ravel()
            m = mag[y:y + CELDA_HOG, x:x + CELDA_HOG].ravel()
            h = np.bincount(b, weights=m, minlength=BINS_HOG)
            salida.append(h / (np.linalg.norm(h) + 1e-6))
    return np.concatenate(salida)


def _defectos_profundos(contorno):
    """Defectos de convexidad: los 'valles' entre el contorno y su casco convexo.
    Los valles hondos son los espacios entre dedos separados. Devuelve la lista de puntos (inicio, fin, fondo)."""
    casco = cv2.convexHull(contorno, returnPoints=False)
    if casco is None or len(casco) < 4:
        return []
    try:
        defectos = cv2.convexityDefects(contorno, casco)
    except cv2.error:              # pasa con contornos que se cruzan a sí mismos
        return []
    if defectos is None:
        return []
    salida = []
    for s, e, f, d in defectos.reshape(-1, 4):    # filas (inicio, fin, punto más hondo, profundidad*256)
        if d / 256.0 > PROF_DEFECTO * TAM_ROI:
            salida.append((tuple(contorno[s][0]), tuple(contorno[e][0]), tuple(contorno[f][0])))
    return salida


def caracteristicas(mascara, contorno, roi_gris):
    """Convierte la mano en un vector de números (lo que usa el clasificador ML).
    Forma del contorno:
      area_rel     área del contorno / área del recorte
      aspecto      ancho / alto del rectángulo que lo encierra
      extension    área del contorno / área de ese rectángulo
      solidez      área del contorno / área del casco convexo (puño ~0.9, mano abierta ~0.6)
      circularidad 4*pi*área / perímetro²  (círculo = 1)
      defectos     nº de valles hondos (≈ dedos separados - 1)
    Momentos de Hu: 7 números que describen la forma y no cambian con traslación, escala ni giro
      (se usa -sign*log10|hu| porque sus valores son diminutos).
    HOG del recorte en gris 64x64: dirección de los bordes por zonas; describe la textura y la
      posición de los dedos aunque la máscara salga incompleta."""
    area = cv2.contourArea(contorno)
    x, y, w, h = cv2.boundingRect(contorno)
    casco = cv2.convexHull(contorno)
    area_casco = max(cv2.contourArea(casco), 1)
    perimetro = max(cv2.arcLength(contorno, True), 1)
    forma = [
        area / (TAM_ROI * TAM_ROI),
        w / max(h, 1),
        area / max(w * h, 1),
        area / area_casco,
        4 * np.pi * area / perimetro ** 2,
        len(_defectos_profundos(contorno)),
    ]
    hu = cv2.HuMoments(cv2.moments(mascara, binaryImage=True)).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    hog = hog_simple(roi_gris)
    return np.concatenate([forma, hu, hog]).astype(np.float32)


def procesar_roi(roi):
    """Atajo que usan el servidor, el entrenamiento y las pruebas: ROI -> (vector, datos para dibujar).
    vector es None si no se encontró un contorno de mano útil."""
    r, mascara, contorno = segmentar_mano(roi)
    if contorno is None:
        return None, (r, mascara, None)
    vector = caracteristicas(mascara, contorno, cv2.cvtColor(r, cv2.COLOR_BGR2GRAY))
    return vector, (r, mascara, contorno)


def dibujar_etapas(roi, mascara, contorno, vector=None, pred_ml=None, alto=200):
    """Tira de 4 paneles para la web:
    ROI mejorada | máscara binaria | contorno (verde) + casco convexo (azul) + defectos (rojo) | números."""
    def panel(img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (alto, alto), interpolation=cv2.INTER_NEAREST)

    dibujo = roi.copy()
    if contorno is not None:
        cv2.drawContours(dibujo, [contorno], -1, (0, 220, 0), 1)
        cv2.drawContours(dibujo, [cv2.convexHull(contorno)], -1, (255, 140, 0), 1)
        for _, _, fondo in _defectos_profundos(contorno):
            cv2.circle(dibujo, fondo, 3, (0, 0, 255), -1)

    texto = np.full((alto, alto, 3), 30, np.uint8)
    lineas = ["SIN CONTORNO"] if vector is None else [
        f"dedos~ {int(vector[5]) + 1 if vector[5] else '0-1'}",
        f"solidez {vector[3]:.2f}",
        f"circular {vector[4]:.2f}",
        f"aspecto {vector[1]:.2f}",
        f"area {vector[0]:.2f}",
    ]
    if pred_ml:
        lineas.append(f"ML: {pred_ml[0]} {pred_ml[1]:.2f}")
    for i, t in enumerate(lineas):
        cv2.putText(texto, t, (8, 46 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    paneles = [panel(roi), panel(mascara), panel(dibujo), texto]
    titulos = ["1 ROI + CLAHE", "2 Otsu + morfologia", "3 contorno + casco", "4 caracteristicas"]
    for p, t in zip(paneles, titulos):
        cv2.rectangle(p, (0, 0), (alto, 18), (0, 0, 0), -1)
        cv2.putText(p, t, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return np.hstack(paneles)
