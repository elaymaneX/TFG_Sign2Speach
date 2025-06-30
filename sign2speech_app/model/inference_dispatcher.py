import numpy as np
import os
import subprocess
import time
import tensorflow as tf


# Ruta base del script actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Modelo TFLite optimizado para EdgeTPU
MODEL_CPU = os.path.join(BASE_DIR, 'model_edgetpu.tflite')

# Rutas en el dispositivo remoto (Coral)
REMOTE_MODEL = "/home/mendel/model_edgetpu.tflite"
REMOTE_SCRIPT = "/home/mendel/remote_inference.py"
REMOTE_INPUT = "/home/mendel/tensor.npy"
REMOTE_OUTPUT = "/home/mendel/result.npy"



def get_remote_device():
    try:
        result = subprocess.run(["mdt", "devices"], capture_output=True, text=True)
        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]

        if not lines:
            raise Exception("No se encontró ningún dispositivo Coral conectado con MDT.")

        hostname = lines[0].split()[0]
        print(f"📡 Coral detectada: {hostname}")
        return hostname

    except Exception as e:
        print("⚠️ No se pudo detectar la Coral con MDT:", e)
        return None


def try_remote_tpu_inference(input_tensor):
    try:
        hostname = get_remote_device()
        if hostname is None:
            raise Exception("No se pudo detectar Coral con MDT")

        print("🔁 Enviando tensor a la Coral TPU con MDT...")
        local_input = "tensor.npy"
        local_output = "."

        # Guardar el tensor localmente
        np.save(local_input, input_tensor)
        time.sleep(1)  # Espera por seguridad

        if not os.path.exists(local_input):
            raise Exception("tensor.npy no se ha creado")
        else:
            print("tensor.npy existe, tamaño:", os.path.getsize(local_input), "bytes")

        # Subir el tensor al dispositivo remoto
        subprocess.run(["mdt", "push", local_input], check=True)

        # Ejecutar script remoto que corre la inferencia
        subprocess.run(["mdt", "exec", "python3", REMOTE_SCRIPT], check=True)

        # Descargar el resultado de vuelta
        subprocess.run(["mdt", "pull", REMOTE_OUTPUT, local_output], check=True)

        # Limpiar archivos temporales en Coral
        subprocess.run(["mdt", "exec", "rm /home/mendel/tensor.npy"], check=True)
        print("done rm 1")
        subprocess.run(["mdt", "exec", "rm /home/mendel/result.npy"], check=True)
        print("done rm 2")

        # Cargar resultado en CPU
        output = np.load("result.npy")

        # Limpiar archivos locales
        os.remove(local_input)
        os.remove("result.npy")

        print("✅ Inference done on remote Edge TPU")
        return output

    except Exception as e:
        print("❌ Error durante la inferencia remota:", e)
        return None

def try_local_cpu_inference(input_tensor):
    try:
        print("💻 Ejecutando inferencia en CPU local...")

        interpreter = tf.lite.Interpreter(model_path=MODEL_CPU)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        print("✅ Inference done on local CPU")
        return output

    except Exception as e:
        print("⚠️ CPU inference failed:", e)
        return None



def run_inference(input_tensor):
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Añade batch dimension

    # Intentar primero en la Coral TPU
    output = try_remote_tpu_inference(input_tensor)
    if output is not None:
        return output

    # Si falla, usar CPU local
    return try_local_cpu_inference(input_tensor)
