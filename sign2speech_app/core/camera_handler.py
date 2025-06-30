import cv2
import numpy as np
import time
import mediapipe as mp
from utils.preprocess import dataPreprocess

class CameraHandler:
    def __init__(self, frames_per_word=35, total_words=3):
        # Inicializa cámara
        self.cap = cv2.VideoCapture(0)

        # Configuración de captura
        self.frames_per_word = frames_per_word  # nº de frames por palabra
        self.total_words = total_words          # nº total de palabras en la secuencia

        # Estado de la captura
        self.current_word = 0
        self.frame_buffer = []       # frames crudos
        self.sequence_data = []      # frames ya preprocesados
        self.is_capturing = False
        self.word_started = False

        # Preprocesamiento
        self.preprocessor = dataPreprocess()

        # Cuenta atrás para cada palabra
        self.countdown_start_time = None
        self.countdown_seconds = 3

        # Inicializa MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Control entre palabras
        self.waiting_between_words = False
        self.wait_start_time = None

        # Último mensaje mostrado (útil para la interfaz)
        self.last_log_message = ""

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def extract_landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        all_landmarks = np.zeros((543, 3), dtype=np.float32)

        def fill_landmarks(landmarks, start_idx):
            if landmarks:
                for i, lm in enumerate(landmarks.landmark):
                    all_landmarks[start_idx + i] = [lm.x, lm.y, lm.z]

        # Rellenar landmarks de pose, cara y manos
        fill_landmarks(results.pose_landmarks, 0)
        fill_landmarks(results.face_landmarks, 33)
        fill_landmarks(results.left_hand_landmarks, 468)
        fill_landmarks(results.right_hand_landmarks, 489)

        return all_landmarks


    def start_sequence_capture(self):
        self.current_word = 0
        self.sequence_data = []
        self.frame_buffer = []
        self.is_capturing = True
        return f"[INFO] Captura de secuencia iniciada..."

    def capture_step(self, frame):
        if not self.is_capturing:
            return None, None, frame

        # Si se han capturado todas las palabras
        if self.current_word >= self.total_words:
            self.is_capturing = False
            return None, "[INFO] Secuencia completa capturada. Lista para inferencia.", frame

        now = time.time()

        # Espera entre palabras
        if self.waiting_between_words:
            if now - self.wait_start_time < 2:
                return None, None, self._add_text_overlay(frame, f"Esperando...")
            else:
                self.waiting_between_words = False
                self.word_started = False

        # Iniciar nueva palabra con cuenta atrás
        if not self.word_started:
            self.word_start_time = now
            self.word_started = True
            log = f"[INFO] Empezando captura palabra {self.current_word + 1}/{self.total_words}..."
            return None, log, self._add_text_overlay(frame, f"Iniciando palabra {self.current_word + 1}...")

        # Mostrar cuenta atrás
        elapsed = now - self.word_start_time
        if elapsed < self.countdown_seconds:
            countdown = self.countdown_seconds - int(elapsed)
            return None, None, self._add_text_overlay(frame, f"{countdown}")

        # Captura de landmarks tras la cuenta atrás
        landmarks = self.extract_landmarks(frame)
        self.frame_buffer.append(landmarks)

        # Si ya tenemos suficientes frames para la palabra
        if len(self.frame_buffer) >= self.frames_per_word:
            video_array = np.array(self.frame_buffer)
            self.frame_buffer = []

            preprocessed = self.preprocessor(video_array)
            self.sequence_data.append(preprocessed)

            self.current_word += 1
            self.word_started = False
            self.waiting_between_words = True
            self.wait_start_time = time.time()

            return preprocessed, f"[INFO] Palabra {self.current_word} capturada y procesada.", self._add_text_overlay(frame, "Captura finalizada")

        return None, None, self._add_text_overlay(frame, f"Capturando palabra {self.current_word + 1}")


    def _add_text_overlay(self, frame, text):
        annotated = frame.copy()
        cv2.putText(annotated, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        return annotated

    def release(self):
        self.cap.release()
        self.holistic.close()


    def get_sequence(self):
        if len(self.sequence_data) == self.total_words:
            return np.stack(self.sequence_data)
        return None

