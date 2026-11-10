# Guía de sustentación

Para que **cualquier integrante** pueda explicar cualquier parte o hacer un cambio pequeño en vivo.

## 1. Arquitectura en una página

```
                         servidor_web.py  (UN solo proceso: dueño de la cámara y del puerto COM)
 ┌───────────────────────────────────────────────────────────────────────────────────────────────┐
 │ hilo_vision()  — por cada cuadro:                                                             │
 │   analizar_cuadro()                                                                           │
 │     1 calidad_cuadro()          procesamiento.py  Laplaciano + canal V    → ¿borroso/oscuro?   │
 │     2 yolo.predict()            runs/senas/weights/best.pt (YOLO11n)       → caja, seña, conf  │
 │     3 recortar_roi()+procesar_roi()  procesamiento.py  CLAHE→YCrCb+Otsu→morfología→contorno    │
 │                                       →casco convexo→defectos→vector (6 forma + 7 Hu + 144 HOG)│
 │     4 clasificador_ml.predecir() modelos/clasificador_ml.joblib (SVM)      → ¿coincide?        │
 │   Confirmador.paso()            5 misma seña válida N cuadros + anti-rebote → letra            │
 │   enviar(letra, "seña")         ─────────── USB serial ───────────►  Pico (main.cpp)          │
 │   agente.avisar_hecho / avisar_no_hecho   → mensaje ✔/✖ en el chat                             │
 │ hilo_lector_serial()            lee "LED=ON VENT=OFF SERVO=ON" → estado (la Pico manda)         │
 │ Flask: /  /video  /etapas.jpg  /api/estado  /api/comando  /api/parametros  /api/metricas      │
 │        /api/agente  → agente.py (function calling o modo básico)                              │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
        ▲ polling cada 300 ms                     web/index.html (HTML+CSS+JS puro)
```

* **Hilos:** visión, lector serial y Flask (un hilo por petición). Comparten `estado`, `registros`,
  `eventos`; por eso se usa `candado` (`threading.Lock`): dos hilos no escriben a la vez.
  `candado_serial` asegura que dos órdenes no se mezclen en el puerto.
* **Un solo punto de salida:** `enviar(letra, origen)` lo usan la visión, los botones y el agente.
  Así el historial siempre sabe quién mandó qué.
* **La Pico es la fuente de verdad:** el estado de los actuadores se toma de lo que *responde* la Pico,
  no de lo que creemos haber mandado.
* **Simulador:** `SimuladorPico` tiene los mismos métodos que `serial.Serial` (`write`, `in_waiting`, `read`)
  y la misma lógica que `main.cpp`. Si la Pico se desconecta, `pasar_a_simulado()` cambia sin caerse.
* **MJPEG (`/video`):** respuesta `multipart/x-mixed-replace`: el servidor manda JPEG tras JPEG en la misma
  conexión y el navegador reemplaza la imagen. Parece video sin librerías.

## 2. Los 4 temas: dónde, qué técnica y por qué

| Tema | Archivo → función | Técnica | Por qué se eligió |
|---|---|---|---|
| **1 Procesamiento** | `procesamiento.py → calidad_cuadro` | Varianza del Laplaciano (nitidez) y media del canal V de HSV (brillo), a 320 px | Barato, no necesita entrenamiento; evita accionar con cuadros movidos u oscuros |
| | `recortar_roi` | ROI = caja de YOLO + 10 % de margen | Procesar solo la mano: más rápido y sin fondo |
| | `segmentar_mano` | CLAHE en Y, YCrCb, Otsu sobre Cr, apertura + cierre, contorno mayor | La piel tiene Cr alto con cualquier luz; Otsu se adapta a cada imagen (dio mejores máscaras que un rango fijo, sobre todo con sobreexposición) |
| | `caracteristicas`, `hog_simple`, `_defectos_profundos` | Área, aspecto, extensión, **solidez**, circularidad, **defectos de convexidad**, **momentos de Hu**, **HOG** | La forma distingue puño (solidez alta) de mano abierta; Hu no cambia con escala/giro; HOG describe bordes de los dedos aunque la máscara falle |
| **2 Machine Learning** | `clasificador_ml.py → entrenar, candidatos, predecir` | `Pipeline(StandardScaler, SVM RBF / Random Forest / KNN)`, se elige el de mejor exactitud en val | Clásico y explicable; escalar es obligatorio porque las características tienen rangos distintos; mismo split de YOLO para no hacer trampa |
| **3 Deep Learning** | `entrenar_senas.py`, `servidor_web.py → analizar_cuadro` | YOLO11n (transfer learning desde `yolo11n.pt`) | Detecta y clasifica en un paso, en tiempo real, con pocos datos |
| **4 Agente IA** | `agente.py → Agente` | Responses API de OpenAI con *function calling* (7 herramientas, incluida `capturar_roi`) + avisos escritos automáticos + modo básico por palabras clave (`_modo_basico`, `_ordenes_en_texto`: parte la frase por «y», comas, «luego» para sacar varias órdenes) | La persona sorda necesita **leer** qué pasó y por qué; las herramientas impiden que el modelo invente estados; el modo básico asegura la demo sin internet |
| **Fusión** | `analizar_cuadro` + `Confirmador` | YOLO conf ≥ `CONF_MIN` **y** ML da ≥ `ML_MIN` a la misma clase, en `FRAMES_CONFIRMAR` cuadros | Dos modelos con información distinta (red profunda vs. forma geométrica) deben estar de acuerdo antes de mover un actuador |

## 3. Parámetros

| Parámetro | Dónde | Valor | Qué hace |
|---|---|---|---|
| `CONF_MIN` | `control_senas.py` | 0.6 | Confianza mínima de YOLO para aceptar la seña. Subirlo = menos falsos positivos, más rechazos |
| `FRAMES_CONFIRMAR` | `control_senas.py` | 5 | Cuadros seguidos con la misma seña válida para enviar la orden |
| `TIEMPO_LIBRE` | `control_senas.py` | 0.8 s | Tiempo sin mano para "desbloquear" y poder repetir la misma seña |
| `ML_MIN` | `servidor_web.py` | 0.50 | Probabilidad mínima que el ML le da a la clase de YOLO. Calibrado: cuando YOLO acierta, el 95 % de los cuadros tiene ≥ 0.78 |
| `CONF_VER` | `servidor_web.py` | 0.25 | YOLO reporta cajas desde 0.25 solo para poder explicar "confianza baja" |
| `CUADROS_AVISO` / `CONF_AVISO` | `servidor_web.py` | 6 / 0.40 | Cuándo el agente escribe "✖ no lo hice": seña rechazada 6 cuadros seguidos con conf ≥ 0.40 |
| `SEG_AVISO_BLOQUEO` | `servidor_web.py` | 2.5 s | Si la seña sigue arriba tras ejecutarse, el agente explica que hay que bajar la mano |
| `NITIDEZ_MIN` | `procesamiento.py` | 20 | Varianza del Laplaciano mínima. Val original ≥ ~130; desenfoque moderado ~66; fuerte ~6 |
| `BRILLO_MIN` | `procesamiento.py` | 50 | Canal V medio mínimo (0-255). Con −40 % de luz el dataset sigue sobre 80 |
| `TAM_ROI` | `procesamiento.py` | 128 | La mano se normaliza a 128×128: las características no dependen de la distancia |
| `AREA_MIN` | `procesamiento.py` | 0.08 | Contorno menor al 8 % del recorte = no hay mano útil |
| `PROF_DEFECTO` | `procesamiento.py` | 0.12 | Profundidad mínima (12 % del lado) para contar un valle entre dedos |
| `CELDA_HOG`, `BINS_HOG` | `procesamiento.py` | 16, 9 | HOG: celdas de 16 px en 64×64 (4×4 celdas) × 9 orientaciones = 144 valores |
| `C=10`, `gamma="scale"` | `clasificador_ml.py` | | SVM RBF: C alto = margen más estricto; gamma = alcance de cada ejemplo |
| `BLOQUE` | `preparar_dataset.py` | 8 | 8 fotos seguidas van juntas a train o a val: fotos vecinas son casi iguales y no deben quedar una en cada lado |
| `VAL_FRAC` | `preparar_dataset.py` | 0.2 | 20 % de los bloques de cada seña van a validación |
| `fliplr` | `entrenar_senas.py` | 0.5 | Espejo horizontal en el 50 % de las imágenes: mano izquierda y derecha. El ML hace lo mismo al aumentar datos |
| `imgsz` | `entrenar_senas.py` | 640 | Tamaño al que YOLO redimensiona la imagen |
| `patience` | `entrenar_senas.py` | 25 | Si en 25 épocas no mejora, para (paró en la 77; la mejor fue la 52) |
| `epochs`, `batch` | `entrenar_senas.py` | 100, 16 | Máximo de épocas; imágenes por paso |

## 4. Preguntas probables (respuesta corta)

1. **¿Qué problema resuelven?** Los asistentes del hogar dependen de la voz; una persona sorda no puede usarlos. Aquí se controla con señas y el sistema responde por escrito y con señales visuales.
2. **¿Dónde está cada tema?** Procesamiento `procesamiento.py`; ML `clasificador_ml.py`; DL YOLO (`entrenar_senas.py`, usado en `analizar_cuadro`); Agente `agente.py`. Se unen en `servidor_web.py → analizar_cuadro`.
3. **¿Por qué es integración real y no 4 demos?** Cada cuadro pasa por los 4: si la calidad falla, si YOLO no ve la mano, si no hay contorno o si el ML no coincide, **no se acciona**. El agente lee esos mismos registros para explicar.
4. **¿Por qué YOLO y no solo un clasificador?** YOLO también **localiza** la mano; sin la caja no sabríamos qué región procesar.
5. **¿Para qué el ML si YOLO ya clasifica?** Es una segunda opinión con información distinta (forma geométrica). Reduce acciones equivocadas (p. ej. con imagen oscura 1→0, giro +15° 4→2).
6. **¿Qué es la solidez?** Área del contorno / área de su casco convexo. Puño ≈ 0.9; mano abierta ≈ 0.6 porque hay huecos entre dedos.
7. **¿Qué es el casco convexo y los defectos?** El casco es la "liga" que envuelve la mano; los defectos son los valles entre la liga y el contorno: los espacios entre dedos.
8. **¿Por qué YCrCb y no RGB?** En RGB la luz cambia los tres canales; en YCrCb la luz va en Y y el color en Cr/Cb, y la piel tiene Cr alto con cualquier iluminación.
9. **¿Qué hace Otsu?** Prueba todos los umbrales y elige el que mejor separa los dos grupos del histograma (piel / no piel). No hay que fijarlo a mano.
10. **¿Apertura y cierre?** Apertura = erosión + dilatación: borra puntitos. Cierre = dilatación + erosión: tapa huecos.
11. **¿Qué es HOG?** Histograma de orientaciones del gradiente: en cada celda cuenta hacia dónde apuntan los bordes. Lo implementamos con Sobel + `cartToPolar` porque OpenCV 5 ya no trae `HOGDescriptor`.
12. **¿Por qué StandardScaler?** SVM y KNN usan distancias; sin escalar, la característica con números más grandes dominaría.
13. **¿Por qué ganó la SVM?** Las tres sacaron 100 % en val (55 manos); se queda la primera en empate. Val es pequeño, por eso también medimos con variaciones.
14. **¿Cómo evitaron hacer trampa en validación?** Split por bloques de 8 fotos consecutivas y por seña; el ML usa exactamente el mismo split de YOLO.
15. **¿Qué significan P, R, mAP50 y mAP50-95?** P: de lo detectado, cuánto era correcto. R: de lo que había, cuánto se encontró. mAP50: precisión media con cajas que se solapan ≥50 %. mAP50-95: promedio exigiendo de 50 % a 95 % de solape.
16. **¿Por qué 5 cuadros y el bloqueo?** Un cuadro suelto puede ser ruido; 5 seguidos (~0.3 s) es intencional. El bloqueo evita que mantener la mano alterne el LED muchas veces.
17. **¿Qué pasa si se desconecta la Pico?** `pasar_a_simulado()` sigue con el simulador desde el último estado; la web muestra "Modo simulado".
18. **¿Y si no hay internet?** El agente usa el modo básico por palabras clave con las mismas herramientas, y los avisos ✔/✖ no usan la API.
19. **¿Cómo sabe el agente que no inventa?** Solo actúa con herramientas; el prompt le exige consultar `obtener_estado` y `metricas_modelo` antes de afirmar.
20. **¿Por qué "apaga todo" no manda T?** `T` apaga si *algo* está encendido pero **enciende** si todo está apagado. El agente mira el estado y manda solo las letras individuales que hagan falta.
21. **¿Por qué nunca `debug=True`?** El recargador de Flask arranca el programa dos veces y abriría la cámara y el COM dos veces.
22. **¿Debilidades?** Con ruido fuerte YOLO baja a ~41 %; con giro o espejo una persona haciendo «led» muy cerca se confunde con «todo» en ambos modelos (error correlacionado); el filtro de nitidez descarta desenfoques fuertes que YOLO aún reconocería (preferimos no accionar con imágenes malas).
23. **¿Qué es la ROI y dónde se saca?** Región de interés: el recorte de la caja de YOLO + 10 % de margen (`procesamiento.recortar_roi`). Se ve en vivo en la web (`/roi.jpg`) y se guarda con el botón o pidiéndoselo al agente (`servidor_web.capturar_roi` → `capturas/`).
24. **¿Cómo entiende el agente sin internet «enciende la luz y el ventilador»?** `_ordenes_en_texto` parte la frase por «y» y comas; cada pedazo busca un verbo (si no tiene, usa el anterior) y un dispositivo, comparando palabras completas (así «¿está apagado?» no se toma como la orden «apaga»).

## 5. Cambios pequeños que pueden pedir en vivo

### a) Cambiar el umbral de confianza
* En vivo: slider **Confianza YOLO mín.** en la web (o escribir al agente "sube la confianza a 0.8").
* En código: `control_senas.py`, línea `CONF_MIN = 0.6` → `0.75`. Reiniciar `servidor_web.py`.
* `ML_MIN` está en `servidor_web.py` (`ML_MIN = 0.50`).

### b) Quitar el ML del flujo
* En vivo: selector **Modo → Solo YOLO**.
* Por defecto: en `servidor_web.py`, dentro de `estado = {...}`, cambiar `"modo": "yolo+ml"` por `"modo": "yolo"`.

### c) Cambiar un pin de la Pico (p. ej. LED a GP17)
1. `pico_senas/src/main.cpp`: `const int PIN_LED = 15;` → `17;`
2. Cerrar el servidor (libera el COM), en PlatformIO **Upload**.
3. Actualizar la tabla de pines en `README.md` y en `web/index.html` (sección "Conexiones").

### d) Cambiar el ángulo del servo (p. ej. 180°)
1. `main.cpp`: `const int SERVO_ENCENDIDO = 90;` → `180;` y **Upload**.
2. Textos: `agente.py → describir_accion` ("servo a 90°"), `web/index.html` (`"Abierta 90°"` y tabla de señas).

### e) Agregar una seña nueva (p. ej. `alarma` → letra `A` → buzzer en GP13)
1. Grabar un video y `python extraer_frames.py video_alarma.mp4`; etiquetar en X-AnyLabeling con `alarma`.
2. Agregar `"alarma"` al final de `CLASES` en `preparar_dataset.py` **y** en `clasificador_ml.py`.
3. `control_senas.py`: `COMANDOS = {..., "alarma": "A"}`.
4. `main.cpp`: `const int PIN_BUZZER = 13;`, `bool alarma = false;`, `pinMode` en `setup`, `else if (c == 'A') alarma = !alarma;`, `digitalWrite` en `aplicar()` y agregar `ALARMA=` a la línea de estado.
5. `servidor_web.py`: en `SimuladorPico.write` agregar la letra `A`; en `hilo_lector_serial` ampliar el patrón; en `enviar` permitir `A`; agregar `"alarma"` a `estado["actuadores"]`.
6. `agente.py`: agregar a `NOMBRE`, `LETRA`, al `enum` de `controlar_actuador` y a `describir_accion`.
7. `web/index.html`: otra tarjeta en "Dispositivos".
8. `python preparar_dataset.py` → `python entrenar_senas.py` → `python clasificador_ml.py` → `python evaluar_sistema.py`.
