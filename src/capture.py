"""Capture, affichage et enregistrement d'images depuis la webcam."""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.camera import CameraError

DEFAULT_SAVE_DIR = Path("captures")
WINDOW_NAME = "Rubik's Cube Vision"
KEY_QUIT = {ord("q"), 27}  # q ou Échap
KEY_CAPTURE = {ord("c"), ord(" ")}


class CaptureError(RuntimeError):
    """Impossible d'obtenir ou d'enregistrer une image valide."""


def is_valid_frame(frame):
    """Vérifie qu'il s'agit d'une image 8 bits non vide (niveaux de gris ou BGR)."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    if frame.dtype != np.uint8:
        return False
    return frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 3)


def describe_frame(frame):
    """Retourne la résolution et le format d'une image valide."""
    if not is_valid_frame(frame):
        raise CaptureError("Image invalide.")
    height, width = frame.shape[:2]
    channels = 1 if frame.ndim == 2 else frame.shape[2]
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "format": "GRAY" if channels == 1 else "BGR",
        "dtype": str(frame.dtype),
    }


def capture_frame(camera, attempts=3):
    """Récupère une image valide depuis une caméra ouverte."""
    for _ in range(max(1, attempts)):
        frame = camera.read()
        if is_valid_frame(frame):
            return frame
    raise CaptureError(f"Aucune image valide après {attempts} tentative(s).")


def save_frame(frame, directory=DEFAULT_SAVE_DIR, name=None):
    """Enregistre l'image en PNG et retourne son chemin."""
    if not is_valid_frame(frame):
        raise CaptureError("Impossible d'enregistrer une image invalide.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if name is None:
        name = datetime.now().strftime("capture_%Y%m%d_%H%M%S_%f")
    path = directory / f"{name}.png"
    if not cv2.imwrite(str(path), frame):
        raise CaptureError(f"Échec de l'écriture de {path}.")
    return path


def run_preview(camera, save_dir=DEFAULT_SAVE_DIR, overlay=None):
    """Affiche le flux en direct ; 'c'/Espace enregistre, 'q'/Échap quitte.

    `overlay(frame)` peut retourner l'image à afficher à la place du flux
    brut ; c'est toujours l'image brute qui est enregistrée.
    Retourne la liste des fichiers enregistrés.
    """
    saved = []
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            try:
                frame = capture_frame(camera)
            except (CaptureError, CameraError) as exc:
                print(f"Erreur de capture : {exc}")
                break

            cv2.imshow(WINDOW_NAME, overlay(frame) if overlay else frame)
            key = cv2.waitKey(1) & 0xFF

            if key in KEY_QUIT or _window_closed():
                break
            if key in KEY_CAPTURE:
                path = save_frame(frame, save_dir)
                saved.append(path)
                info = describe_frame(frame)
                print(f"Image enregistrée : {path} "
                      f"({info['width']}x{info['height']}, {info['format']})")
    finally:
        cv2.destroyWindow(WINDOW_NAME)
        cv2.waitKey(1)  # laisse Qt traiter la fermeture de la fenêtre
    return saved


def _window_closed():
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
