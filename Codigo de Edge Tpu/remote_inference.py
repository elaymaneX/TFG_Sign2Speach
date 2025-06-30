import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate


# Inicializa el intérprete TFLite con soporte para Edge TPU
interpreter = Interpreter(
    model_path="model_edgetpu.tflite",
    experimental_delegates=[load_delegate("libedgetpu.so.1")]
)
interpreter.allocate_tensors()


# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Cargar el tensor de entrada desde archivo
input_tensor = np.load("tensor.npy").astype(np.float32)

# Asignar el tensor al intérprete
interpreter.set_tensor(input_details[0]["index"], input_tensor)


# Ejecutar la inferencia
interpreter.invoke()

# Obtener la salida del modelo
output_tensor = interpreter.get_tensor(output_details[0]["index"])


# Guardar resultado en un archivo para recuperar desde el host
np.save("result.npy", output_tensor)

print("✅ Inference done remotely.")
