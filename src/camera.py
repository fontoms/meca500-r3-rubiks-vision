"""Connexion à la webcam (Logitech C920) via OpenCV."""

import sys

import cv2

DEFAULT_INDEX = 0
if sys.platform.startswith("linux"):
    DEFAULT_BACKEND = cv2.CAP_V4L2
elif sys.platform == "win32":
    DEFAULT_BACKEND = cv2.CAP_MSMF
else:
    DEFAULT_BACKEND = cv2.CAP_ANY
# Sous WSL2 (usbipd), le format YUYV par défaut sature le lien USB et provoque
# des "select() timeout" : le flux MJPG compressé est indispensable.
DEFAULT_FOURCC = "MJPG"
# Le lien usbipd perd parfois les premières images : on réessaie avant d'abandonner.
DEFAULT_WARMUP_ATTEMPTS = 3


class CameraError(RuntimeError):
    """La caméra est introuvable ou ne peut pas être initialisée."""


class Camera:
    """Webcam OpenCV utilisable comme gestionnaire de contexte."""

    def __init__(self, index=DEFAULT_INDEX, backend=DEFAULT_BACKEND,
                 fourcc=DEFAULT_FOURCC, warmup_attempts=DEFAULT_WARMUP_ATTEMPTS):
        if not isinstance(index, int) or index < 0:
            raise ValueError(f"Index de caméra invalide : {index!r}")
        self.index = index
        self.backend = backend
        self.fourcc = fourcc
        self.warmup_attempts = max(1, warmup_attempts)
        self._capture = None

    @property
    def is_opened(self):
        return self._capture is not None and self._capture.isOpened()

    def open(self):
        """Ouvre le flux vidéo et vérifie que la caméra répond."""
        if self.is_opened:
            return self
        try:
            capture = cv2.VideoCapture(self.index, self.backend)
        except cv2.error as exc:
            raise CameraError(
                f"Erreur OpenCV à l'ouverture de la caméra {self.index} : {exc}"
            ) from exc

        if not capture.isOpened():
            capture.release()
            raise CameraError(f"Caméra {self.index} introuvable ou occupée.")

        if self.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*self.fourcc))

        # grab() force la caméra à produire une image sans la décoder :
        # c'est le seul moyen fiable de vérifier qu'elle est réellement prête.
        if not any(capture.grab() for _ in range(self.warmup_attempts)):
            capture.release()
            raise CameraError(
                f"Caméra {self.index} ouverte mais ne fournit aucune image."
            )

        self._capture = capture
        return self

    def release(self):
        """Libère la caméra (sans effet si elle n'est pas ouverte)."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read(self):
        """Lit une image du flux ; retourne None si la lecture échoue."""
        if not self.is_opened:
            raise CameraError("La caméra n'est pas ouverte.")
        ok, frame = self._capture.read()
        return frame if ok else None

    def info(self):
        """Retourne les propriétés du flux ouvert."""
        if not self.is_opened:
            raise CameraError("La caméra n'est pas ouverte.")
        cap = self._capture
        return {
            "index": self.index,
            "backend": cap.getBackendName(),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
        }

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def list_cameras(max_index=4, backend=DEFAULT_BACKEND, fourcc=DEFAULT_FOURCC,
                 warmup_attempts=DEFAULT_WARMUP_ATTEMPTS):
    """Retourne les index des caméras utilisables parmi 0..max_index-1."""
    available = []
    for index in range(max_index):
        try:
            with Camera(index, backend, fourcc, warmup_attempts):
                available.append(index)
        except CameraError:
            continue
    return available
