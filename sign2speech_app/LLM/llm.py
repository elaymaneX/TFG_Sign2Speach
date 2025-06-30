import requests

# URL del endpoint del Space de Hugging Face que genera frases a partir de palabras
HF_SPACE_URL = "https://aelamraxx-sentence-generator-api.hf.space/translate"


def generate_sentence_from_words(words: list[str]) -> str:
    """
    Envía una lista de palabras al Space de Hugging Face y devuelve la frase generada.

    Args:
        words (list[str]): Lista de palabras predichas.

    Returns:
        str: Frase generada por el modelo LLM.
    """
    try:
        # Petición POST al endpoint de Hugging Face
        response = requests.post(
            HF_SPACE_URL,
            json={"words": words},
            headers={"Content-Type": "application/json"}
        )
        # Lanza excepción si el código de respuesta no es 200
        response.raise_for_status()

        # Extrae la frase generada desde la respuesta JSON
        return response.json().get("sentence", "")

    except Exception as e:
        # Si ocurre algún error de red o respuesta inválida
        print("❌ Error al comunicar con Hugging Face Space:", e)
        return ""

