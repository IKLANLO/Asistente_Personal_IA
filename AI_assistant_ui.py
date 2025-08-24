import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import pyttsx3
import time
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav

# --------------------------
# Funciones auxiliares de voz
# --------------------------

def hablar(texto: str, rate: int = 170, volume: float = 1.0):
    if not texto:
        return
    try:
        engine = pyttsx3.init()
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

        engine.say(texto)
        engine.runAndWait()
        engine.stop()
        time.sleep(0.05)
    except Exception as e:
        st.error(f"Error en TTS: {e}")

# --------------------------
# Inicialización de IA y memoria
# --------------------------

llm = OllamaLLM(model="mistral")
if "historial" not in st.session_state:
    st.session_state.historial = ChatMessageHistory()

prompt = PromptTemplate(
    variables=["historial", "pregunta"],
    template="Historial: {historial}\nUsuario: {pregunta}\nRespuesta:"
)

def procesar_respuesta(pregunta):
    historial_de_chat = "\n".join(
        [f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.historial.messages]
    )
    respuesta = llm.invoke(prompt.format(historial=historial_de_chat, pregunta=pregunta))
    st.session_state.historial.add_user_message(pregunta)
    st.session_state.historial.add_ai_message(respuesta)
    return respuesta

# --------------------------
# Grabar audio solo al pulsar 🎤
# --------------------------

def grabar_y_reconocer(segundos=5):
    fs = 16000  # frecuencia de muestreo
    st.info(f"🎙️ Grabando {segundos} segundos... Habla ahora")
    audio = sd.rec(int(segundos * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()

    # guardar a archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        wav.write(tmp.name, fs, audio)
        tmp_path = tmp.name

    r = sr.Recognizer()
    with sr.AudioFile(tmp_path) as source:
        audio_data = r.record(source)
        try:
            texto = r.recognize_google(audio_data, language="es-ES")
            return texto
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None

# --------------------------
# UI con Streamlit
# --------------------------

st.set_page_config(page_title="Asistente IA", page_icon="🤖")
st.title("🤖 Asistente de IA con voz")

st.write("Haz tu pregunta por **texto o micrófono**.")

col1, col2 = st.columns([4,1])
with col1:
    pregunta_texto = st.text_input("Escribe tu pregunta aquí:")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    grabar = st.button("🎤")

# --- Lógica de envío ---
if pregunta_texto and st.button("Enviar"):
    st.chat_message("user").write(pregunta_texto)
    respuesta = procesar_respuesta(pregunta_texto)
    st.chat_message("assistant").write(respuesta)
    hablar(respuesta)

if grabar:
    pregunta = grabar_y_reconocer()
    if pregunta:
        st.success(f"Te he entendido: {pregunta}")
        # 🚀 Enviar automáticamente la transcripción
        st.chat_message("user").write(pregunta)
        respuesta = procesar_respuesta(pregunta)
        st.chat_message("assistant").write(respuesta)
        hablar(respuesta)
    else:
        st.error("No se pudo reconocer el audio")

# Mostrar historial
st.subheader("Historial de conversación")
for msg in st.session_state.historial.messages:
    role = "🧑 Usuario" if msg.type == "human" else "🤖 Asistente"
    st.write(f"**{role}:** {msg.content}")
