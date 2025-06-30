from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QTextEdit, QSizePolicy, QSpacerItem
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import QTimer, Qt
from core.camera_handler import CameraHandler
from LLM.llm import generate_sentence_from_words
from TTS.tts import speak_text
import cv2


class RoundedLabel(QLabel):
    """
    QLabel personalizada con esquinas redondeadas usando hoja de estilo (CSS).
    Ideal para mostrar vídeo, texto o mensajes sobre fondo oscuro estilizado.
    """
    def __init__(self, radius=24, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius

        # Estilo visual: fondo oscuro y bordes redondeados
        self.setStyleSheet(f"""
            background-color: #222;
            border-radius: {self.radius}px;
        """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign2Speech")
        self.setFixedSize(1050, 600)

        # Fondo pastel azul
        pastel_blue = "#f3f0fc"
        self.setStyleSheet(f"background-color: {pastel_blue};")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # --- Panel Izquierdo: Botones y Log ---
        self.capture_button = QPushButton("Capturar Secuencia")
        self.Adios = QPushButton("Adios")  # Botón para cerrar app

        # Estilo personalizado para ambos botones
        for btn, color in zip([self.Adios, self.capture_button], ["#D22C19", "#43A047"]):
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    font-weight: bold;
                    font-size: 15px;
                    border-radius: 18px;
                    padding: 8px 0px;
                }}
                QPushButton:hover {{
                    background-color: #333;
                }}
            """)
            btn.setFixedHeight(38)
            btn.setMinimumWidth(180)
            btn.setCursor(Qt.PointingHandCursor)

        # Caja de logs
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(160)
        self.log_box.setMinimumWidth(180)
        self.log_box.setStyleSheet("""
            QTextEdit {
                border-radius: 12px;
                border: 2px solid #807f7f;
                font-size: 13px;
                padding: 4px;
                background: #f6f6f6;
            }
        """)

        # Organización vertical del panel izquierdo
        stack_layout = QVBoxLayout()
        stack_layout.addWidget(self.Adios)
        stack_layout.addSpacing(12)
        stack_layout.addWidget(self.capture_button)
        stack_layout.addSpacing(16)
        stack_layout.addWidget(self.log_box)

        stack_widget = QWidget()
        stack_widget.setLayout(stack_layout)

        left_layout = QVBoxLayout()
        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        left_layout.addWidget(stack_widget, alignment=Qt.AlignCenter)
        left_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        left_container = QWidget()
        left_container.setLayout(left_layout)
        left_container.setFixedWidth(220)

        # --- Vista de Cámara ---
        self.video_label = RoundedLabel(radius=24)
        self.video_label.setFixedSize(700, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Cámara no iniciada")

        # --- Layout Principal (Izquierda + Cámara) ---
        main_layout = QHBoxLayout()
        main_layout.addWidget(left_container)
        main_layout.addSpacing(18)
        main_layout.addWidget(self.video_label)
        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 0)
        main_layout.setStretch(2, 1)

        # --- Logo abajo a la derecha ---
        from PyQt5.QtGui import QPixmap
        logo_label = QLabel()
        logo_pixmap = QPixmap("./Pics/logo.png")  # Ajusta a tu ruta
        logo_pixmap = logo_pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        logo_label.setStyleSheet("margin: 0 0px 0px 0; background: transparent;")

        # Layout global que incluye todo
        global_layout = QVBoxLayout()
        global_layout.addLayout(main_layout)

        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch(1)
        bottom_bar.addWidget(logo_label)
        global_layout.addLayout(bottom_bar)

        self.central_widget.setLayout(global_layout)

        # --- Configurar cámara y refresco de frames ---
        self.camera = CameraHandler()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Actualiza cada 30ms

        # Conectar botones
        self.capture_button.clicked.connect(self.start_sequence_capture)
        self.Adios.clicked.connect(self.close)

    # 📄 Agrega mensaje al log (pantalla + consola)
    def log_message(self, message):
        self.log_box.append(message)
        print(message)

    # 🔁 Actualiza cada frame: cámara, inferencia, texto y audio
    def update_frame(self):
        frame = self.camera.read_frame()
        if frame is None:
            return

        if self.camera.is_capturing:
            preprocessed, log_message, annotated_frame = self.camera.capture_step(frame)

            if log_message and log_message != self.camera.last_log_message:
                self.log_message(log_message)
                self.camera.last_log_message = log_message

            # Si ya terminó la captura de todas las palabras
            if not self.camera.is_capturing and len(self.camera.sequence_data) == self.camera.total_words:
                sequence = self.camera.get_sequence()
                if sequence is not None:
                    self.log_message("[INFO] Ejecutando inferencia...")
                    print("[DEBUG] sequence shape:", sequence.shape)

                    from model.inference import predict_words
                    palabras = predict_words(sequence)

                    frase = generate_sentence_from_words(palabras)
                    self.log_message(f"[RESULTADO] {' '.join(palabras)}")
                    self.log_message(f"[FRASE] {frase}")

                    speak_text(frase)

        else:
            annotated_frame = frame

        # Mostrar frame en interfaz
        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    # ▶️ Iniciar captura de secuencia
    def start_sequence_capture(self):
        message = self.camera.start_sequence_capture()
        self.log_message(message)

    # ❌ Al cerrar ventana, liberar cámara
    def closeEvent(self, event):
        self.camera.release()
        super().closeEvent(event)
