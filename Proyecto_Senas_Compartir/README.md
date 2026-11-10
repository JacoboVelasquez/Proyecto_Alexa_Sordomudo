# Asistente domótico controlado por señas — un «Alexa» para personas sordas

**Visión Artificial · Universidad EIA** · Jacobo Velásquez Durango · Federico Pérez Cortés · Samuel Sánchez · Jerónimo Arias Velásquez

## Problema
Los asistentes del hogar (Alexa, Google Home) se controlan con la voz y responden con sonido. Las personas sordas
o con discapacidad auditiva y del habla quedan por fuera de una tecnología pensada para dar autonomía.

## Solución
La persona hace una **seña con la mano** frente a la cámara; el sistema la reconoce con visión artificial, la
verifica y enciende o apaga dispositivos reales (luz, ventilador, cerradura con servo) a través de una
Raspberry Pi Pico. Todo lo que pasa se le comunica **por escrito y con señales visuales**: un agente de IA escribe
en el chat cada acción realizada (✔) o no realizada (✖) y **por qué**, y responde preguntas o recibe órdenes por texto.

## Temas integrados: 4 — nivel MUY ALTO, nota máxima 5.0

Los 4 temas están en **el mismo flujo**: cada cuadro de la cámara pasa por todos, y si alguno falla no se acciona.

```
cámara ─► [Procesamiento] calidad ─► [Deep Learning] YOLO ─► [Procesamiento] ROI, piel, contorno, características
       ─► [Machine Learning] ¿coincide con YOLO? ─► [Decisión] N cuadros + anti-rebote ─► Pico ─► LED / ventilador / servo
       ─► [Agente IA] escribe qué hizo o no y por qué · responde preguntas · controla por texto
```

| Tema | Archivo → función | Qué aporta al flujo |
|---|---|---|
| **Procesamiento básico/intermedio** | `procesamiento.py` → `calidad_cuadro`, `recortar_roi`, `segmentar_mano`, `caracteristicas`, `hog_simple`, `dibujar_etapas` | Descarta cuadros borrosos u oscuros; recorta la mano; CLAHE, YCrCb + Otsu, apertura/cierre, contorno, casco convexo, defectos de convexidad; vector de 157 características (forma, Hu, HOG) |
| **Machine Learning** | `clasificador_ml.py` → `entrenar`, `predecir` | SVM / Random Forest / KNN (scikit-learn) con esas características; **verifica** la seña de YOLO: si no coinciden no se acciona |
| **Deep Learning** | `entrenar_senas.py`, `runs/senas/weights/best.pt`; uso en `servidor_web.py → analizar_cuadro` | YOLO11n localiza la mano y clasifica la seña con una confianza |
| **Agente de IA (texto)** | `agente.py` → `Agente` | Conversa con la persona y **hace lo que le escribe** («enciende la luz y el ventilador», «abre la puerta»); avisos escritos ✔/✖ con el motivo; *function calling* con 7 herramientas (estado, controlar actuador, explicar decisión, historial, métricas, ajustar parámetros, **capturar la ROI de la mano**); modo básico sin internet que entiende varias órdenes en una frase |

**Extras (no suben el techo, sí la calidad):** mecatrónica (Pico + LED, ventilador con transistor, servo) y
desarrollo web (interfaz de demostración y monitoreo en `web/index.html`).

### Diferencias con la propuesta inicial
| Propuesta | Implementado | Por qué |
|---|---|---|
| ESP32 por WiFi/HTTP | Raspberry Pi Pico por USB serial | No depende de la red del salón (la demo no puede caerse por WiFi). La arquitectura permite volver a ESP32/HTTP cambiando **solo** la función `enviar()` de `servidor_web.py` |
| 6 gestos encender/apagar | 4 señas que **alternan** (led, ventilador, servo, todo) | Menos clases = modelo más robusto con pocos datos (339 fotos). El agente sí puede "encender" o "apagar" explícitamente porque consulta el estado antes de mandar |
| Indicador de confirmación (LED/buzzer) | Confirmación visual en la web (destello grande + aviso escrito) | Mismo objetivo de accesibilidad, más información (dice *qué* y *por qué*) |

## Estructura
```
proyecto_vision_artificial/
├── control_senas.py        Versión original solo YOLO (sigue funcionando sola). Comparte COMANDOS, CONF_MIN, Camara, abrir_pico
├── procesamiento.py        Tema 1: calidad, segmentación de la mano, contornos y características
├── clasificador_ml.py      Tema 2: entrena y carga el clasificador clásico
├── entrenar_senas.py       Tema 3: entrenamiento de YOLO11n
├── agente.py               Tema 4: agente con herramientas + avisos + modo básico
├── servidor_web.py         Une todo: visión + decisión + serial + Flask
├── evaluar_sistema.py      Pruebas: YOLO vs ML vs fusión, negativas y variaciones; crea demo_val.mp4
├── web/index.html          Interfaz (HTML + CSS + JS en un solo archivo, sin internet)
├── extraer_frames.py       Paso 1: video → fotos
├── preparar_dataset.py     Paso 2: JSON de X-AnyLabeling → formato YOLO + split train/val
├── images/                 339 fotos + JSON de X-AnyLabeling (68 negativas)
├── dataset_senas/          Split train/val ya hecho (265 / 74 fotos)
├── runs/senas/             Entrenamiento de YOLO: weights/best.pt, results.csv, gráficas
├── modelos/                clasificador_ml.joblib, metricas_ml.json, pruebas.json (generados)
├── demo_val.mp4            Dato de ejemplo: fotos de validación como video (para probar sin cámara)
├── pico_senas/             Proyecto PlatformIO de la Pico (platformio.ini + src/main.cpp)
├── requirements.txt  ·  .env.ejemplo  ·  .gitignore
└── GUIA_SUSTENTACION.md    Arquitectura, técnicas, parámetros, preguntas y cambios en vivo
```

## Instalación
1. Python 3.12+ (probado con 3.14). Con GPU NVIDIA instala **primero** torch con CUDA:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
2. ```
   pip install -r requirements.txt
   ```
3. Agente con OpenAI (opcional): copia `.env.ejemplo` como `.env` y pon `OPENAI_API_KEY` (y si quieres `OPENAI_MODEL`;
   por defecto el de la plantilla del curso). Si no hay `.env` aquí, se usa `../agentIA/agentIA/.env`.
   **Sin clave o sin internet el agente funciona igual en modo básico.**

## Ejecución
```
python servidor_web.py                           # cámara 0, busca la Pico sola
python servidor_web.py --camara 1 --puerto COM5  # cámara USB externa y puerto fijo
python servidor_web.py --camara demo_val.mp4     # sin cámara: video de ejemplo en bucle
python servidor_web.py --simular                 # sin Pico: simulador idéntico a main.cpp
```
Abrir **http://localhost:5000**. La web muestra "Pico conectada (COMx)" o "Modo simulado".

* Sin Pico conectada, el servidor pasa solo a modo simulado. Si la Pico se desconecta en plena demo, sigue en simulado sin caerse.
* `control_senas.py` (solo YOLO, ventana de OpenCV) sigue disponible: `python control_senas.py --camara demo_val.mp4`.
* **Cierra el monitor serial de PlatformIO y `control_senas.py` antes de correr el servidor**: Windows no deja abrir el COM dos veces.

### La interfaz
* **En vivo:** video con la caja de YOLO y la predicción del ML; **ROI de la mano en vivo** con botón «📷 Guardar ROI»
  (se guarda en `capturas/` y aparece en el chat; también se le puede pedir al agente «saca el ROI de mi mano»); flujo de 6 etapas en verde/rojo con el **motivo** de cada
  cuadro; barra de confirmación; tira de etapas del procesamiento (ROI, máscara, contorno + casco + defectos, números);
  probabilidades del ML por clase; tarjetas de dispositivos con botones; chat con el agente; ajustes en vivo
  (`CONF_MIN`, `FRAMES_CONFIRMAR`, `ML_MIN`, modo yolo / yolo+ml); historial de órdenes (seña / agente / manual).
* **Modelos y pruebas:** métricas de YOLO y del ML, matriz de confusión, tabla de variaciones, datos por seña, gráficas.
* **Cómo funciona:** problema, diagrama de los 4 temas, señas → letras → acciones, conexiones de la Pico.

## Las 4 señas
| Clase | Letra | Qué hace la Pico |
|---|---|---|
| `led` | `L` | Luz apagada ↔ encendida |
| `ventilador` | `V` | Ventilador apagado ↔ encendido |
| `servo` | `S` | Cerradura: servo 0° ↔ 90° |
| `todo` | `T` | Si algo está encendido apaga todo; si todo está apagado enciende todo |
| — | `?` | Solo responde el estado: `LED=ON VENT=OFF SERVO=ON` |

## Modelos y resultados
**YOLO11n** (`runs/senas/weights/best.pt`, mejor época 52 de 77): P 0.928 · R 0.987 · mAP50 0.993 · mAP50-95 0.885.

**Clasificador ML** (`python clasificador_ml.py`, ~15 s): 157 características por mano, train 843 (con espejo y ±25 % de brillo)
/ val 55. SVM RBF, Random Forest y KNN sacan **100 %** en validación; se usa la SVM. Ojo: el val es pequeño y las fotos
de un mismo video se parecen, por eso las pruebas con variaciones (abajo) son la medida más honesta. En el Random Forest
el HOG pesa 84 %, la forma 11 % y los momentos de Hu 5 %.

**Pruebas** (`python evaluar_sistema.py`, ~1 min con GPU) sobre las 74 fotos de validación (58 con seña, 16 negativas), por cuadro:

| Variación | YOLO solo | ML (caja real) | Fusión YOLO+ML | Acciones equivocadas (YOLO → fusión) | Falsos positivos en 16 negativas |
|---|---|---|---|---|---|
| original | 98.3 % | 94.8 % | 94.8 % | 1 → 1 | 1 → 1 |
| oscura −40 % | 98.3 % | 94.8 % | 96.6 % | 1 → 0 | 1 → 1 |
| clara +40 % | 89.7 % | 82.8 % | 77.6 % | 1 → 1 | 1 → 1 |
| desenfoque moderado | 96.6 % | 94.8 % | 93.1 % | 1 → 1 | 0 → 0 |
| desenfoque fuerte | 87.9 % | 94.8 % | 0 % (filtro de calidad) | 2 → 0 | 0 → 0 |
| ruido | 41.4 % | 72.4 % | 34.5 % | 2 → 0 | 0 → 0 |
| giro +15° | 91.4 % | 62.1 % | 75.9 % | 4 → 2 | 1 → 1 |
| giro −15° | 82.8 % | 72.4 % | 75.9 % | 5 → 5 | 1 → 0 |
| espejo | 91.4 % | 94.8 % | 87.9 % | 5 → 5 | 1 → 1 |

Lectura honesta: la fusión **cambia recall por seguridad** — rechaza algunos cuadros correctos pero reduce las acciones
equivocadas (oscura, ruido, giro +15°, desenfoque fuerte). Donde no ayuda (espejo, giro −15°) es porque un bloque de
«led» hecho muy cerca de la cámara por otra persona lo confunden **los dos** modelos con «todo»: más fotos de ese caso
lo arreglarían. En vivo, además, se exige la misma seña en 5 cuadros seguidos, lo que filtra los aciertos/errores sueltos.
Umbrales calibrados con estas pruebas: `ML_MIN = 0.50`, `NITIDEZ_MIN = 20`, `BRILLO_MIN = 50`.

## Datos de ejemplo y cómo reentrenar
* `demo_val.mp4` (generado por `evaluar_sistema.py`) permite toda la demo sin cámara ni Pico:
  `python servidor_web.py --camara demo_val.mp4 --simular`.
* Reentrenar desde cero:
  1. `python extraer_frames.py video_x.mp4` (un video por seña y uno **sin seña**).
  2. Etiquetar `images/` en X-AnyLabeling: rectángulo ajustado a la mano con la etiqueta en minúscula; foto sin seña = JSON sin cajas.
  3. `python preparar_dataset.py` (regenera `dataset_senas/` y `data.yaml` con la ruta de este computador).
  4. `python entrenar_senas.py` → `runs/senas/weights/best.pt`. *Out of memory* → `batch=8`; pérdidas `nan` en la GTX 1650 → `amp=False`.
  5. `python clasificador_ml.py` → `modelos/clasificador_ml.joblib`.
  6. `python evaluar_sistema.py` → `modelos/pruebas.json` y `demo_val.mp4`.

## La Pico (PlatformIO)
1. Abre `pico_senas/` en VS Code con PlatformIO y dale **Upload**.
2. Prueba sin Python: monitor serial, escribe `L`, `V`, `S`, `T` y mira que responda `LED=ON VENT=OFF SERVO=OFF`.
3. **Cierra el monitor serial** antes de correr el servidor.

| Pico | Conexión |
|---|---|
| GP15 | resistencia 220 Ω → LED → GND |
| GP14 | base/gate de un transistor o MOSFET (2N2222 con 1 kΩ, o IRLZ44N) que enciende el ventilador; diodo 1N4007 en paralelo con el ventilador. **Nunca el ventilador directo al pin** |
| GP16 | señal del servo; servo alimentado con 5 V (VBUS o fuente externa) |
| GND | GND común con la fuente del ventilador/servo |

## Problemas comunes
| Síntoma | Solución |
|---|---|
| "No se pudo abrir COMx… ¿monitor serial?" | Cierra el monitor de PlatformIO o `control_senas.py`; el servidor sigue en modo simulado mientras tanto |
| Video negro con "No se pudo abrir la cámara" | Prueba `--camara 1`; cierra otras apps que usen la cámara. El servidor reintenta cada 3 s |
| El agente dice "(Modo básico, sin internet)" | Falta `OPENAI_API_KEY`, la clave es inválida o no hay internet. Todo sigue funcionando con palabras clave |
| "Clasificador ML no entrenado: modo solo YOLO" | `python clasificador_ml.py` |
| Dispara solo | Sube `CONF_MIN` o `FRAMES_CONFIRMAR` (sliders en la web) o agrega fotos negativas |
| Muchos ✖ "ML no coincide" | Mejor luz y fondo sin tonos de piel; o baja `ML_MIN` |
| Muy lento (pocos FPS) | Verifica que diga GPU en el encabezado; si dice CPU instala torch con CUDA |

## Guion de demostración (5–7 min)
1. **Problema (30 s):** pestaña *Cómo funciona*: por qué un Alexa no sirve a una persona sorda.
2. **Flujo y métricas (1 min):** diagrama de 4 temas; pestaña *Modelos y pruebas*: YOLO (mAP50 0.993), ML (comparación de 3 modelos), tabla de variaciones.
3. **Cada seña en vivo (2 min):** pestaña *En vivo*; hacer `led`, `ventilador`, `servo`, `todo` mostrando la tira de etapas
   (máscara, contorno, casco, defectos), la barra 1/5→5/5, el destello ✔ y el aviso escrito del agente. Mantener la mano para
   mostrar "bloqueada: baja la mano".
4. **Caso rechazado (1 min):** tapar parte de la mano, poca luz o una seña ambigua → el agente escribe ✖ con el motivo
   ("YOLO dijo X pero el ML dijo Y"); preguntar "¿Por qué no se encendió?".
5. **Agente controlando (1 min):** "Hola", "¿Qué está encendido?", "Enciende la luz y el ventilador", repetirlo (no manda nada),
   "Abre la puerta", "Saca el ROI de mi mano" (aparece la foto en el chat), "Apaga todo".
6. **Variaciones (1 min):** alejarse/acercarse, cambiar la luz; mover el slider de `CONF_MIN` o pasar a modo solo YOLO y ver la diferencia.
7. **Cierre (30 s):** historial de órdenes (seña / agente / manual) y "Haz un resumen de la sesión".

Plan B sin cámara/Pico: `python servidor_web.py --camara demo_val.mp4 --simular`.
