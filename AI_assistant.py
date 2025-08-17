import speech_recognition as sr
import pyttsx3
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Inicializamos el modelo de IA
llm = OllamaLLM(model="mistral")

# Inicializamos la memoria de chat
historial = ChatMessageHistory()

# Inicializamos el motor de texto a voz
motor = pyttsx3.init()
motor.setProperty('rate', 170)  # Ajustamos la velocidad de habla (por defecto es 200)

# Inicializamos el reconocedor de voz
reconocedor = sr.Recognizer()

# función para pronunciar el texto indicado
import re

def hablar(texto):
    if texto and texto.strip():
        # 1. Quitar emojis y caracteres no ASCII
        limpio = re.sub(r'[^\x00-\x7F]+', ' ', texto)

        # 2. Quitar saltos de línea extra
        limpio = limpio.replace("\n", " ").replace("\r", " ").strip()

        print(f"🔊 Texto que se va a pronunciar: {limpio}")
        motor.say(limpio)
        motor.runAndWait()
    else:
        print("⚠️ No hay texto para pronunciar.")

# función para escuchar y reconocer el audio
def escuchar():
  with sr.Microphone() as fuente:
    print("Escuchando...")
    reconocedor.adjust_for_ambient_noise(fuente)  # Reducimos el ruido ambiente
    # audio = reconocedor.listen(fuente, timeout=20, phrase_time_limit=20)  # Escuchamos el audio
    audio = reconocedor.listen(fuente)  # Escuchamos el audio
    try:
      texto = reconocedor.recognize_google(audio, language='es-ES')  # Reconocemos el audio
      print(f"Te he entendido lo siguiente: {texto}")
      return texto.lower()  # Devolvemos el texto en minúsculas
    
    except sr.UnknownValueError:
      print("No se pudo reconocer el audio")
      return None
    
    except sr.RequestError as e:
      print(f"Error al conectar con el servicio de reconocimiento: {e}")
      return None
    
# Prompt para la IA
prompt = PromptTemplate(
  variables=["historial", "pregunta"],
  template="Historial: {historial}\nUsuario: {pregunta}\nRespuesta:"
)

# Función para procesar la respuesta de la IA
def procesar_respuesta(pregunta):
  # Recuperamos el historial de chat
  historial_de_chat = "\n".join([f"{msg.type.capitalize()}:{msg.content}" for msg in historial.messages])  # Convertimos el historial a un string

  # Generamos la respuesta
  respuesta = llm.invoke(prompt.format(historial=historial_de_chat, pregunta=pregunta))

  # Guardamos la pregunta y respuesta en el historial
  historial.add_user_message(pregunta)
  historial.add_ai_message(respuesta)

  return respuesta

# Bucle principal del asistente
hablar("¡Hola!, soy tu asistente de IA. ¿En qué puedo ayudarte hoy?")

while True:
  pregunta = escuchar()  # Escuchamos la pregunta del usuario
  
  if pregunta is None:
    hablar("Lo siento, no he podido entenderte. ¿Podrías repetirlo?")
    continue  # Si no se reconoce la pregunta, volvemos al inicio del bucle
  
  if "salir" in pregunta:
    hablar("¡Hasta luego!")
    break
  
  respuesta = procesar_respuesta(pregunta)  # Procesamos la respuesta de la IA
  print(f"Respuesta IA: {respuesta}")
  hablar(respuesta)  # Pronunciamos la respuesta de la IA

