import numpy as np

#  Parámetros globales de preprocesamiento
INPUT_SIZE = 64       # Longitud objetivo de cada secuencia (en frames)
GAP = 8               # Máximo número de frames consecutivos vacíos que se pueden interpolar

#  Índices de landmarks seleccionados (pose, cara y manos)
LANDMARK_IDX = [0, 9, 11, 13, 14, 17, 117, 118, 119, 199, 346, 347, 348] + list(range(468, 543))

class dataPreprocess:
    def __init__(self, input_size=INPUT_SIZE, max_gap=GAP, landmark_idxs=LANDMARK_IDX):
        self.input_size = input_size        # Frames por secuencia
        self.max_gap = max_gap              # Máximo tramo interpolable
        self.landmark_idxs = landmark_idxs  # Landmarks a conservar

    def interpolate_missing(self, seq):
        """
        Interpola secciones vacías (todo a 0.0) si son suficientemente cortas.
        """
        seq = seq.copy()
        mask = np.all(seq == 0.0, axis=(1, 2))
        i = 0
        while i < len(seq):
            if mask[i]:
                start = i
                while i < len(seq) and mask[i]:
                    i += 1
                end = i
                gap = end - start

                # No interpolar si está al inicio/final o si el gap es muy largo
                if start == 0 or end == len(seq) or gap > self.max_gap:
                    continue

                # Interpolación lineal
                for j in range(gap):
                    alpha = (j + 1) / (gap + 1)
                    seq[start + j] = (1 - alpha) * seq[start - 1] + alpha * seq[end]
            else:
                i += 1
        return seq

    def pad(self, video, pad_left, pad_right):
        """
        Añade padding replicando el primer y último frame.
        """
        return np.concatenate([
            np.repeat(video[:1], pad_left, axis=0),
            video,
            np.repeat(video[-1:], pad_right, axis=0)
        ], axis=0)

    def __call__(self, video):
        """
        Preprocesamiento completo: interpolación, selección, padding y remuestreo.
        """
        # Interpolar frames vacíos
        video = self.interpolate_missing(video)

        # Filtrar landmarks si se especificó
        if self.landmark_idxs is not None:
            video = video[:, self.landmark_idxs, :]

        T, L, D = video.shape
        N = self.input_size

        if T < N:
            # Si la secuencia es más corta, rellenar con padding
            pad_total = N - T
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            video = self.pad(video, pad_left, pad_right)
            return video.astype(np.float32)

        if T == N:
            return video.astype(np.float32)

        # Si la secuencia es más larga, repetir y hacer media
        repeat_factor = (N * N) // T
        video = np.repeat(video, repeats=repeat_factor, axis=0)

        T = video.shape[0]
        excess = T % N
        if excess > 0:
            pad_total = N - excess
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            video = self.pad(video, pad_left, pad_right)

        video = video.reshape(N, -1, L, D)
        return video.mean(axis=1).astype(np.float32)
