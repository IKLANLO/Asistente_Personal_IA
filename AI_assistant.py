import speech_recognition as sr
import pyttsx3
import time
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# --------------------------
# Funciones auxiliares de voz
# --------------------------

def hablar(texto: str, rate: int = 170, volume: float = 1.0):
    if not texto:
        print("⚠️ No hay texto para pronunciar.")
        return
    try:
        engine = pyttsx3.init()  # motor fresco en cada llamada
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)

        # seleccionar voz en español si existe
        sel = None
        for v in engine.getProperty("voices"):
            name = (getattr(v, "name", "") or "").lower()
            vid  = (getattr(v, "id", "")   or "").lower()
            langs = "".join(getattr(v, "languages", []) or []).lower()
            if any(s in (name + " " + vid + " " + langs)
                   for s in ["spanish", "es-es", "es_mx", "es-", "es_"]):
                sel = v.id
                break
        if sel:
            engine.setProperty("voice", sel)

        print(f"🔊 Texto que se va a pronunciar: {texto!r}")
        engine.stop()
        engine.say(texto)
        engine.runAndWait()
        engine.stop()
        time.sleep(0.05)
    except Exception as e:
        print("⚠️ Error en TTS:", e)

# --------------------------
# Inicialización de IA y memoria
# --------------------------

llm = OllamaLLM(model="mistral")
historial = ChatMessageHistory()
reconocedor = sr.Recognizer()

# --------------------------
# Función de escucha
# --------------------------

def escuchar():
    with sr.Microphone() as fuente:
        print("Escuchando...")
        reconocedor.adjust_for_ambient_noise(fuente, duration=1)
        try:
            audio = reconocedor.listen(fuente, timeout=5, phrase_time_limit=20)
            texto = reconocedor.recognize_google(audio, language='es-ES')
            print(f"Te he entendido lo siguiente: {texto}")
            return texto.lower()
        except sr.WaitTimeoutError:
            print("⏳ No se detectó voz a tiempo.")
            return None
        except sr.UnknownValueError:
            print("No se pudo reconocer el audio")
            return None
        except sr.RequestError as e:
            print(f"Error al conectar con el servicio de reconocimiento: {e}")
            return None

# --------------------------
# Prompt y procesamiento
# --------------------------

prompt = PromptTemplate(
    variables=["historial", "pregunta"],
    template="Historial: {historial}\nUsuario: {pregunta}\nRespuesta:"
)

def procesar_respuesta(pregunta):
    historial_de_chat = "\n".join([f"{msg.type.capitalize()}:{msg.content}" for msg in historial.messages])
    respuesta = llm.invoke(prompt.format(historial=historial_de_chat, pregunta=pregunta))
    historial.add_user_message(pregunta)
    historial.add_ai_message(respuesta)
    print("Respuesta IA:", respuesta)
    return respuesta

# --------------------------
# Programa principal
# --------------------------

if __name__ == "__main__":
    hablar("¡Hola!, soy tu asistente de IA. ¿En qué puedo ayudarte hoy?")
    while True:
        pregunta = escuchar()
        if not pregunta:
            continue

        if "salir" in pregunta:
            hablar("Hasta luego!")
            break

        hablar("Pensando...")
        respuesta = procesar_respuesta(pregunta)
        hablar(respuesta)
