"""
TEMA 4 — AGENTE DE IA (texto).

Pensado para personas sordas: TODO es escrito. El agente:
  1. Escribe solo en el chat cada vez que el sistema HACE o NO HACE una acción, y por qué
     (avisar_hecho / avisar_no_hecho). Son plantillas de texto: instantáneas y sin internet.
  2. Responde preguntas y controla los actuadores usando herramientas (function calling de la
     Responses API de OpenAI, mismo estilo que la plantilla del profesor en ../agentIA/agentIA).
  3. Si no hay API key o no hay internet, sigue funcionando en "modo básico" (palabras clave).

El agente NUNCA toca la Pico directamente: solo actúa a través de las herramientas, que usan
las mismas funciones del servidor (enviar, estado, registros...).
"""
import json
import os
import re
import threading
import unicodedata
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

BASE = Path(__file__).resolve().parent
# .env de esta carpeta; si no existe, el de la plantilla del profesor
load_dotenv(BASE / ".env") or load_dotenv(BASE.parent / "agentIA" / "agentIA" / ".env")
MODELO = os.getenv("OPENAI_MODEL") or "gpt-5.6-luna"     # el mismo de la plantilla del profesor
MAX_VUELTAS = 5          # máximo de rondas modelo -> herramientas -> modelo por pregunta
MAX_HISTORIAL = 20       # mensajes de conversación que se recuerdan

NOMBRE = {"led": "la luz (LED)", "ventilador": "el ventilador", "servo": "la cerradura (servo)",
          "todo": "todos los dispositivos"}
LETRA = {"led": "L", "ventilador": "V", "servo": "S"}


@dataclass
class Sistema:
    """Lo que el agente puede usar del servidor (se le pasa desde servidor_web.py)."""
    estado: Callable
    enviar: Callable
    registros: Callable
    eventos: Callable
    metricas: Callable
    ajustar: Callable
    capturar_roi: Callable


INSTRUCCIONES = """Eres el asistente escrito de un sistema domótico controlado por señas de la mano,
hecho para personas sordas (un "Alexa" sin voz). Responde SIEMPRE en español, corto y claro:
quien te lee es una persona sorda o un evaluador; no uses relleno.

Cómo funciona el sistema (flujo de 4 temas integrados):
1. Procesamiento de imágenes: se mide la calidad del cuadro (nitidez = varianza del Laplaciano, brillo = canal V);
   en la región de la mano se aplica CLAHE, YCrCb + Otsu (piel), morfología, contorno, casco convexo y defectos,
   y se sacan características (solidez, circularidad, defectos ≈ dedos, momentos de Hu, HOG).
2. Deep Learning: YOLO11n detecta la mano y dice qué seña es, con una confianza (0-1).
3. Machine Learning: un clasificador clásico (scikit-learn) usa esas características y debe estar de acuerdo con YOLO.
4. Decisión: la misma seña válida en varios cuadros seguidos se confirma y se envía una letra por USB a una
   Raspberry Pi Pico. Después de enviar, la seña queda bloqueada hasta que la persona baje la mano.
5. Tú (agente) explicas decisiones, consultas resultados y controlas los actuadores por texto.

Señas: led -> L (luz), ventilador -> V, servo -> S (cerradura 0°/90°), todo -> T (si algo está encendido apaga todo,
si todo está apagado enciende todo). La Pico solo ALTERNA; la herramienta controlar_actuador ya sabe encender/apagar.

Métricas: P (precisión) = de lo que detectó, cuánto era correcto; R (recall) = de lo que había, cuánto encontró;
mAP50 = calidad de detección con cajas que se solapan ≥50 %; mAP50-95 = promedio exigiendo más solape (más estricto);
exactitud ML = porcentaje de manos de validación bien clasificadas.

Reglas:
- Conversa con naturalidad y amabilidad: saluda, pregunta qué necesita la persona y ofrece ayuda.
- Si la persona pide encender, apagar, abrir o cerrar algo, HAZLO con controlar_actuador (una llamada por actuador)
  y confirma por escrito cómo quedó todo.
- Si pide la ROI, un recorte o una foto de la mano, usa capturar_roi y di qué seña se vio.
- Usa obtener_estado antes de afirmar cómo está un actuador.
- Para "¿por qué no se encendió/no hizo nada?" usa explicar_ultima_decision y di el motivo concreto con números.
- No inventes métricas: usa metricas_modelo.
- Si te piden algo que no es de este sistema, di amablemente que solo ayudas con el sistema domótico.
"""

HERRAMIENTAS = [
    {"type": "function", "name": "obtener_estado",
     "description": "Estado actual: actuadores encendidos/apagados, conexión con la Pico, modo, parámetros y FPS.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"type": "function", "name": "controlar_actuador",
     "description": "Enciende, apaga o alterna un actuador. Solo envía órdenes si hace falta.",
     "parameters": {"type": "object", "properties": {
         "actuador": {"type": "string", "enum": ["led", "ventilador", "servo", "todo"]},
         "accion": {"type": "string", "enum": ["encender", "apagar", "alternar"]}},
         "required": ["actuador", "accion"], "additionalProperties": False}},
    {"type": "function", "name": "explicar_ultima_decision",
     "description": "Últimos registros de decisión del flujo de visión (YOLO, ML, calidad, motivo de aceptar/rechazar).",
     "parameters": {"type": "object", "properties": {"n": {"type": "integer", "description": "cuántos (1-30)"}},
                    "required": [], "additionalProperties": False}},
    {"type": "function", "name": "consultar_historial",
     "description": "Últimas órdenes enviadas a la Pico: hora, origen (seña/agente/manual), comando y estado resultante.",
     "parameters": {"type": "object", "properties": {"n": {"type": "integer", "description": "cuántos (1-50)"}},
                    "required": [], "additionalProperties": False}},
    {"type": "function", "name": "metricas_modelo",
     "description": "Métricas de YOLO, del clasificador ML y de las pruebas ante variaciones.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"type": "function", "name": "ajustar_parametros",
     "description": "Cambia umbrales: conf_min (0.30-0.95), frames_confirmar (1-15), ml_min (0-1), modo (yolo | yolo+ml).",
     "parameters": {"type": "object", "properties": {
         "conf_min": {"type": "number"}, "frames_confirmar": {"type": "integer"},
         "ml_min": {"type": "number"}, "modo": {"type": "string", "enum": ["yolo", "yolo+ml"]}},
         "required": [], "additionalProperties": False}},
    {"type": "function", "name": "capturar_roi",
     "description": "Guarda la ROI (recorte de la mano que encontró YOLO) del último cuadro y la muestra en el chat.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
]

# Palabras que entiende el modo básico (sin internet). Se comparan PALABRAS completas, no pedazos:
# así "¿está apagado?" no se confunde con la orden "apaga".
PALABRAS_ACTUADOR = [({"ventilador", "abanico"}, "ventilador"),
                     ({"servo", "cerradura", "puerta", "chapa"}, "servo"),
                     ({"todo", "todos", "todas"}, "todo"),
                     ({"led", "luz", "luces", "bombillo", "lampara", "foco"}, "led")]
PALABRAS_ACCION = [({"enciende", "encienda", "encender", "prende", "prenda", "prender", "activa", "activar",
                     "abre", "abra", "abrir"}, "encender"),
                   ({"apaga", "apague", "apagar", "desactiva", "desactivar", "cierra", "cierre", "cerrar"}, "apagar"),
                   ({"cambia", "cambiar", "alterna", "alternar"}, "alternar")]


def sin_tildes(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")


def describir_accion(letra, despues):
    """Texto humano de lo que hizo una letra, según el estado final que respondió la Pico."""
    if letra == "L":
        return "encendí la luz (LED)" if despues["led"] else "apagué la luz (LED)"
    if letra == "V":
        return "encendí el ventilador" if despues["ventilador"] else "apagué el ventilador"
    if letra == "S":
        return "abrí la cerradura (servo a 90°)" if despues["servo"] else "cerré la cerradura (servo a 0°)"
    if letra == "T":
        return "encendí todo" if any(despues.values()) else "apagué todo"
    return "consulté el estado"


def texto_estado(a):
    return (f"Luz {'encendida' if a['led'] else 'apagada'}, ventilador {'encendido' if a['ventilador'] else 'apagado'}, "
            f"cerradura {'abierta' if a['servo'] else 'cerrada'}.")


def mayuscula(texto):
    return texto[:1].upper() + texto[1:]          # a diferencia de capitalize(), no baja "LED" a "led"


class Agente:
    def __init__(self, sistema: Sistema):
        self.s = sistema
        self.candado = threading.Lock()
        self.chat = deque(maxlen=60)        # lo que ve la web: {id, hora, tipo, texto, acciones}
        self.siguiente_id = 1
        self.conversacion = []              # historial para el modelo de lenguaje
        self.cliente = None
        clave = os.getenv("OPENAI_API_KEY")
        if clave:
            try:
                from openai import OpenAI
                self.cliente = OpenAI(api_key=clave)
            except Exception as e:
                print(f"[AVISO] No se pudo crear el cliente de OpenAI ({e}). Agente en modo básico.")
        print(f"Agente: {'OpenAI ' + MODELO if self.cliente else 'modo básico (sin API key)'}")
        self._escribir("agente", "¡Hola! Soy tu asistente. Te escribiré cada vez que haga (✔) o no haga (✖) una acción "
                                 "y por qué. También puedes escribirme lo que quieras: «enciende la luz y el ventilador», "
                                 "«¿qué está encendido?», «saca el ROI de mi mano».")

    # ------------------------------------------------ mensajes del chat
    def _escribir(self, tipo, texto, acciones=None, imagen=None, modo=None):
        """tipo: usuario | agente | hecho | no_hecho.  imagen: url de una ROI guardada (se ve en el chat).
        modo: 'básico' cuando respondió sin OpenAI (la web lo muestra pequeño junto a la hora)."""
        with self.candado:
            self.chat.append({"id": self.siguiente_id, "hora": datetime.now().strftime("%H:%M:%S"), "tipo": tipo,
                              "texto": texto, "acciones": acciones or [], "imagen": imagen, "modo": modo})
            self.siguiente_id += 1

    def chat_reciente(self):
        with self.candado:
            return list(self.chat)

    # ------------------------------------------------ avisos automáticos (sin API)
    def avisar_hecho(self, letra, origen, antes, despues, reg, params):
        """✔ Se realizó una acción: qué se hizo y por qué."""
        accion = mayuscula(describir_accion(letra, despues))
        if origen == "seña" and reg:
            y, ml = reg["yolo"], reg["ml"]
            porque = f"YOLO vio la seña «{y['clase']}» ({y['conf']:.2f})"
            if params["modo"] == "yolo+ml" and ml:
                porque += f" y el clasificador ML la confirmó ({ml['prob_yolo']:.2f})"
            else:
                porque += " (modo solo YOLO, sin verificación ML)"
            porque += f" durante {params['frames_confirmar']} cuadros seguidos"
        else:
            porque = "porque se presionó el botón manual"
        if antes == despues:
            accion = f"Envié «{letra}» pero la Pico no reportó cambio"
        self._escribir("hecho", f"{accion}: {porque}. {texto_estado(despues)}", [f"envió {letra}"])

    def avisar_no_hecho(self, tipo, reg, params):
        """✖ YOLO vio una seña pero NO se accionó: decir el motivo concreto."""
        y = reg["yolo"]
        clase = y["clase"] if y else "?"
        orden = f"«{clase}» ({NOMBRE.get(clase, clase)})"
        if tipo == "bloqueada":
            texto = (f"No repetí {orden}: ya la ejecuté y sigues mostrando la misma seña. "
                     f"Baja la mano un segundo para poder repetirla.")
        elif reg["etapa_falla"] == "calidad":
            texto = f"No hice nada: {reg['motivo']}. Mejora la luz o deja la mano quieta."
        elif reg["etapa_falla"] == "yolo":
            texto = (f"No ejecuté {orden}: YOLO la vio con confianza {y['conf']:.2f}, menor que el mínimo "
                     f"{params['conf_min']:.2f}. Acerca la mano o haz la seña más clara.")
        elif reg["etapa_falla"] == "procesamiento":
            texto = (f"No ejecuté {orden}: no pude separar la mano del fondo (sin contorno de piel). "
                     f"Revisa la luz o el fondo.")
        elif reg["etapa_falla"] == "ml":
            ml = reg["ml"]
            if ml and ml["clase"] != clase:
                texto = (f"No ejecuté {orden}: YOLO dijo «{clase}» ({y['conf']:.2f}) pero el clasificador ML "
                         f"dijo «{ml['clase']}» ({ml['prob']:.2f}). Como no coinciden, por seguridad no actué.")
            else:
                texto = (f"No ejecuté {orden}: el clasificador ML solo le dio {ml['prob_yolo'] if ml else 0:.2f} "
                         f"a «{clase}», menos que el mínimo {params['ml_min']:.2f}.")
        else:
            texto = f"No ejecuté {orden}: {reg['motivo']}."
        self._escribir("no_hecho", texto)

    # ------------------------------------------------ herramientas
    def obtener_estado(self):
        e = self.s.estado()
        return {"actuadores": e["actuadores"], "resumen": texto_estado(e["actuadores"]), "conexion": e["conexion"],
                "modo": e["parametros"]["modo"], "parametros": e["parametros"], "fps": e["fps"],
                "dispositivo": e["dispositivo"]}

    def controlar_actuador(self, actuador, accion, origen="agente"):
        """La Pico solo alterna: se mira el estado y se manda la letra SOLO si hace falta.
        'encender/apagar todo' manda las letras individuales que falten (no T, que podría hacer lo contrario)."""
        if actuador not in NOMBRE or accion not in ("encender", "apagar", "alternar"):
            return {"error": "actuador o acción inválidos"}
        a = self.s.estado()["actuadores"]
        if actuador == "todo":
            letras = ["T"] if accion == "alternar" else \
                [LETRA[k] for k in ("led", "ventilador", "servo") if a[k] != (accion == "encender")]
        elif accion == "alternar" or a[actuador] != (accion == "encender"):
            letras = [LETRA[actuador]]
        else:
            letras = []
        for l in letras:
            self.s.enviar(l, origen)
        final = self.s.estado()["actuadores"]
        if not letras:
            mensaje = f"No envié nada: {NOMBRE[actuador]} ya estaba así. {texto_estado(final)}"
        else:
            mensaje = f"Envié {', '.join(letras)}. {texto_estado(final)}"
        return {"letras_enviadas": letras, "mensaje": mensaje, "actuadores": final}

    def explicar_ultima_decision(self, n=10):
        n = max(1, min(30, int(n or 10)))
        regs = self.s.registros(200)
        # Lo útil para explicar son los cuadros donde se vio una mano (los "sin mano" se resumen)
        con_mano = [r for r in regs if r["yolo"]][-n:]
        resumen = []
        for r in con_mano:
            item = {"hora": r["hora"], "yolo": f"{r['yolo']['clase']} {r['yolo']['conf']:.2f}",
                    "motivo": r["motivo"], "valido": r["valido"], "enviado": r["enviado"]}
            if r["ml"]:
                item["ml"] = f"{r['ml']['clase']} {r['ml']['prob']:.2f} (a la clase de YOLO: {r['ml']['prob_yolo']:.2f})"
            if r["calidad"]:
                item["calidad"] = r["calidad"]
            resumen.append(item)
        e = self.s.estado()
        return {"parametros": e["parametros"], "cuadros_con_mano": resumen,
                "cuadros_recientes_sin_mano": sum(1 for r in regs[-30:] if not r["yolo"]),
                "ultimo_motivo": regs[-1]["motivo"] if regs else "sin datos"}

    def consultar_historial(self, n=10):
        n = max(1, min(50, int(n or 10)))
        return {"eventos": self.s.eventos(n)}

    def metricas_modelo(self):
        m = self.s.metricas()
        ml = m.get("ml") or {}
        return {"yolo_mejor_epoca": (m.get("yolo") or {}).get("mejor"),
                "ml": {"modelo": ml.get("modelo_elegido"), "exactitud_val": ml.get("exactitud_val"),
                       "comparacion": ml.get("comparacion")},
                "pruebas": (m.get("pruebas") or {}).get("resumen"), "conteo_dataset": m.get("conteo")}

    def ajustar_parametros(self, **cambios):
        p, errores = self.s.ajustar(cambios)
        return {"parametros": p, "errores": errores}

    def capturar_roi(self):
        """Guarda en capturas/ el recorte de la mano (ROI) del último cuadro donde YOLO vio una mano."""
        return self.s.capturar_roi()

    def mostrar_roi(self, r):
        """Cuando la persona pulsa «Guardar ROI» en la web, el agente lo escribe en el chat con la imagen."""
        if "error" in r:
            self._escribir("no_hecho", r["error"])
        else:
            self._escribir("agente", r["mensaje"], imagen=r["url"])

    def _ejecutar(self, nombre, argumentos):
        funciones = {"obtener_estado": self.obtener_estado, "controlar_actuador": self.controlar_actuador,
                     "explicar_ultima_decision": self.explicar_ultima_decision,
                     "consultar_historial": self.consultar_historial, "metricas_modelo": self.metricas_modelo,
                     "ajustar_parametros": self.ajustar_parametros, "capturar_roi": self.capturar_roi}
        if nombre not in funciones:
            return {"error": f"herramienta desconocida {nombre}"}
        try:
            return funciones[nombre](**argumentos)
        except Exception as e:
            return {"error": str(e)}

    def _anotar(self, resultado, extra):
        """Lo que una herramienta hizo y se muestra bajo la respuesta: letras enviadas e imagen de la ROI."""
        if not isinstance(resultado, dict):
            return
        extra["acciones"] += [f"envió {l}" for l in resultado.get("letras_enviadas", [])]
        if resultado.get("url"):
            extra["imagen"] = resultado["url"]

    # ------------------------------------------------ conversación con la persona
    def preguntar(self, mensaje):
        self._escribir("usuario", mensaje)
        extra, modo = {"acciones": [], "imagen": None}, "openai"
        try:
            if not self.cliente:
                raise RuntimeError("sin API key")
            texto = self._preguntar_openai(mensaje, extra)
        except Exception as e:
            print(f"[AVISO] Agente en modo básico: {e.__class__.__name__}: {str(e)[:120]}")
            if e.__class__.__name__ in ("AuthenticationError", "PermissionDeniedError", "NotFoundError"):
                self.cliente = None          # clave o modelo inválidos: no tiene sentido reintentar en cada mensaje
            modo = "básico"
            texto = self._modo_basico(mensaje, extra)
        self._escribir("agente", texto, extra["acciones"], extra["imagen"], modo)
        return {"respuesta": texto, "acciones": extra["acciones"], "imagen": extra["imagen"], "modo": modo}

    def _preguntar_openai(self, mensaje, extra):
        """Function calling con la Responses API:
        1. Se manda la conversación + la lista de herramientas.
        2. Si el modelo pide herramientas (items 'function_call'), se ejecutan aquí y se le devuelven
           los resultados como 'function_call_output' con el mismo call_id.
        3. Se repite hasta que el modelo responda texto (máximo MAX_VUELTAS)."""
        entrada = self.conversacion + [{"role": "user", "content": mensaje}]
        for _ in range(MAX_VUELTAS):
            r = self.cliente.responses.create(model=MODELO, instructions=INSTRUCCIONES,
                                              input=entrada, tools=HERRAMIENTAS)
            llamadas = [item for item in r.output if item.type == "function_call"]
            if not llamadas:
                texto = r.output_text.strip()
                self.conversacion += [{"role": "user", "content": mensaje}, {"role": "assistant", "content": texto}]
                self.conversacion = self.conversacion[-MAX_HISTORIAL:]
                return texto
            entrada += r.output                                   # lo que pidió el modelo
            for item in llamadas:
                resultado = self._ejecutar(item.name, json.loads(item.arguments or "{}"))
                self._anotar(resultado, extra)
                entrada.append({"type": "function_call_output", "call_id": item.call_id,
                                "output": json.dumps(resultado, ensure_ascii=False)})
        return "No pude terminar la respuesta (demasiados pasos). Intenta preguntar de otra forma."

    def _ordenes_en_texto(self, t):
        """Saca TODAS las órdenes de una frase: «enciende la luz y el ventilador, y cierra la puerta»
        -> [(led, encender), (ventilador, encender), (servo, apagar)].
        Se parte la frase por «y», comas, «luego», «también»; si un pedazo no trae verbo, usa el anterior."""
        ordenes, accion = [], None
        for pedazo in re.split(r",|\by\b|\be\b|\bluego\b|\bdespues\b|\btambien\b", t):
            palabras = set(re.findall(r"[a-zñ]+", pedazo))
            accion = next((a for p, a in PALABRAS_ACCION if palabras & p), accion)
            for p, actuador in PALABRAS_ACTUADOR:
                if palabras & p and accion:
                    ordenes.append((actuador, accion))
        return ordenes

    def _modo_basico(self, mensaje, extra):
        """Sin internet o sin API key: entiende frases comunes por palabras clave y usa las MISMAS herramientas."""
        t = sin_tildes(mensaje)
        palabras = set(re.findall(r"[a-zñ]+", t))

        if "por que" in t or "porque" in palabras:
            d = self.explicar_ultima_decision(5)
            if not d["cuadros_con_mano"]:
                return f"No he visto ninguna mano hace rato. Último motivo: {d['ultimo_motivo']}."
            u = d["cuadros_con_mano"][-1]
            extra_ml = f", ML {u['ml']}" if "ml" in u else ""
            return f"En el último cuadro con mano ({u['hora']}) YOLO vio {u['yolo']}{extra_ml}. Motivo: {u['motivo']}."

        quiere_roi = bool(palabras & {"roi", "recorte", "recorta", "captura", "capturar", "foto", "fotografia"})
        ordenes = self._ordenes_en_texto(t)
        if quiere_roi and not ordenes:
            r = self.capturar_roi()
            self._anotar(r, extra)
            return r.get("mensaje") or r.get("error")
        if ordenes:
            mensajes = []
            if quiere_roi:                                    # «enciende la luz y saca el roi»: hace las dos cosas
                r = self.capturar_roi()
                self._anotar(r, extra)
                mensajes.append(r.get("mensaje", r.get("error", "")).rstrip("."))
            for actuador, accion in ordenes:
                r = self.controlar_actuador(actuador, accion)
                self._anotar(r, extra)
                if r.get("error"):
                    mensajes.append(r["error"])
                elif r["letras_enviadas"]:
                    verbos = {"encender": "Abrí" if actuador == "servo" else "Encendí",
                              "apagar": "Cerré" if actuador == "servo" else "Apagué", "alternar": "Cambié"}
                    mensajes.append(f"{verbos[accion]} {NOMBRE[actuador]}")
                else:
                    mensajes.append(f"{mayuscula(NOMBRE[actuador])} ya estaba así")
            return f"Listo. {'. '.join(mensajes)}. Ahora: {texto_estado(self.s.estado()['actuadores'])}"

        if palabras & {"hola", "buenas", "buenos", "hey", "saludos"}:
            return (f"¡Hola! Soy tu asistente. Puedes hacer una seña o escribirme qué quieres, por ejemplo "
                    f"«enciende la luz y el ventilador». Ahora mismo: {texto_estado(self.s.estado()['actuadores'])}")
        if palabras & {"gracias", "genial", "perfecto", "excelente"}:
            return "¡Con gusto! ¿Quieres que encienda o apague algo más?"
        if "que puedes" in t or palabras & {"ayuda", "ayudame", "opciones", "comandos"} or "quien eres" in t:
            return ("Puedo: encender, apagar o cambiar la luz, el ventilador y la cerradura (también varios a la vez); "
                    "decirte qué está encendido; explicarte por qué una seña no funcionó; guardar la ROI de tu mano; "
                    "mostrar el historial y las métricas de los modelos.")
        if "resumen" in palabras or "historial" in palabras:
            ev = self.consultar_historial(50)["eventos"]
            if not ev:
                return "Todavía no se ha enviado ninguna orden en esta sesión."
            por_origen = {}
            for e in ev:
                por_origen[e["origen"]] = por_origen.get(e["origen"], 0) + 1
            ultimos = "; ".join(f"{e['hora']} {e['origen']} {e['comando']}" for e in ev[-5:])
            return f"{len(ev)} órdenes ({por_origen}). Últimas: {ultimos}. {self.obtener_estado()['resumen']}"
        if palabras & {"metrica", "metricas", "modelo", "modelos", "precision", "exactitud"} or "que tan bueno" in t:
            m = self.metricas_modelo()
            y, ml = m["yolo_mejor_epoca"] or {}, m["ml"]
            return (f"YOLO (época {y.get('epoca')}): P {y.get('precision')}, R {y.get('recall')}, "
                    f"mAP50 {y.get('map50')}, mAP50-95 {y.get('map50_95')}. ML: {ml['modelo']} con exactitud "
                    f"{ml['exactitud_val']} en validación.")
        if "como funciona" in t or palabras & {"explica", "explicame"}:
            return ("La cámara ve tu mano; se revisa la calidad de la imagen; YOLO detecta la seña; se recorta la mano "
                    "(ROI), se segmenta la piel y se saca el contorno; un clasificador ML confirma la seña; si coinciden "
                    "varios cuadros seguidos se envía una letra a la Pico, que cambia la luz, el ventilador o la cerradura.")
        if palabras & {"estado", "encendido", "encendida", "apagado", "apagada", "abierta", "cerrada"} or "como esta" in t:
            return self.obtener_estado()["resumen"]
        return ("No entendí. Prueba: «enciende la luz y el ventilador», «apaga todo», «¿qué está encendido?», "
                "«saca el ROI de mi mano», «¿por qué no se encendió?», «historial».")

    def reiniciar(self):
        self.conversacion = []
        with self.candado:
            self.chat.clear()
        self._escribir("agente", "Conversación reiniciada. ¿Qué quieres que haga?")
