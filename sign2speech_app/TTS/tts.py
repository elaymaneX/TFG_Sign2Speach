import pyttsx3

def speak_text(text: str):
    """
    Convierte una cadena de texto en audio usando pyttsx3 (síntesis de voz en local).

    Args:
        text (str): Frase a pronunciar en voz alta.
    """
    # Inicializa el motor de voz
    engine = pyttsx3.init()

    # Obtener y reducir la velocidad del habla (por defecto suele ser muy rápida)
    rate = engine.getProperty("rate")
    engine.setProperty("rate", int(rate * 0.7))  # Velocidad al 70%

    # Enviar texto al motor de voz
    engine.say(text)
    engine.runAndWait()
