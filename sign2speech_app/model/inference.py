import numpy as np
import os
import subprocess
import json


# Ruta al archivo de mapeo de índices a palabras (índice → gloss)
DICT_PATH = os.path.join(os.path.dirname(__file__), "ord2sign.json")

# Cargar diccionario JSON con las clases predichas
with open(DICT_PATH, "r") as f:
    ord2sign = json.load(f)


# Importa la función que se encarga de decidir si se usa la Coral o la CPU
from .inference_dispatcher import run_inference

def predict_words(X: np.ndarray) -> list[str]:
    """
    Ejecuta inferencia sobre una secuencia de N palabras preprocesadas.

    Args:
        X (np.ndarray): Array con shape (N, 64, 88, 3)

    Returns:
        list[str]: Lista de palabras predichas (glosses)
    """
    assert X.ndim == 4 and X.shape[1:] == (64, 88, 3), "Input shape must be (N, 64, 88, 3)"
    X = X.astype(np.float32)

    predictions = []
    for i in range(X.shape[0]):
        output = run_inference(X[i])  # Vector de probabilidades de clase
        pred_idx = int(np.argmax(output))  # Índice con mayor probabilidad
        predictions.append(ord2sign[str(pred_idx)])  # Traducir a palabra

    return predictions

