"""Calibration des couleurs sous l'éclairage réel.

L'utilisateur présente successivement six faces ; le sticker central de
chacune fournit la couleur de référence (BGR) correspondante. Les références
sont enregistrées en JSON et servent ensuite à la reconnaissance.
"""

import json
from pathlib import Path

import cv2

from src.camera import CameraError
from src.capture import WINDOW_NAME, CaptureError, capture_frame
from src.color_recognition import COLORS, lab_distance, median_bgr
from src.detection import detect_face, draw_detection
from src.grid import extract_stickers

DEFAULT_CONFIG_PATH = Path("config/colors.json")
CENTER_INDEX = 4
MIN_REFERENCE_DISTANCE = 20  # en dessous, deux références risquent d'être confondues
COLOR_NAMES_FR = {
    "white": "BLANC", "yellow": "JAUNE", "red": "ROUGE",
    "orange": "ORANGE", "green": "VERT", "blue": "BLEU",
}
KEY_VALIDATE = {ord(" "), ord("c")}
KEY_CANCEL = {ord("q"), 27}


class CalibrationError(RuntimeError):
    """Fichier de calibration illisible ou invalide."""


def center_reference(stickers):
    """Couleur BGR médiane du sticker central d'une face."""
    return [round(float(v)) for v in median_bgr(stickers.patches[CENTER_INDEX])]


def validate_references(references):
    """Vérifie qu'il y a une couleur BGR valide pour chacune des six couleurs."""
    if not isinstance(references, dict) or set(references) != set(COLORS):
        raise CalibrationError(f"Les six couleurs sont attendues : {COLORS}")
    for color, bgr in references.items():
        if (not isinstance(bgr, (list, tuple)) or len(bgr) != 3
                or not all(isinstance(v, (int, float)) and 0 <= v <= 255 for v in bgr)):
            raise CalibrationError(f"Valeur BGR invalide pour {color} : {bgr!r}")
    return references


def close_pairs(references, min_distance=MIN_REFERENCE_DISTANCE):
    """Paires de couleurs de référence trop proches l'une de l'autre."""
    colors = list(references)
    return [(a, b) for i, a in enumerate(colors) for b in colors[i + 1:]
            if lab_distance(references[a], references[b]) < min_distance]


def save_calibration(references, path=DEFAULT_CONFIG_PATH):
    """Enregistre les références en JSON et retourne le chemin."""
    validate_references(references)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {color: [int(round(v)) for v in references[color]] for color in COLORS}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def load_calibration(path=DEFAULT_CONFIG_PATH):
    """Charge les références ; None si le fichier n'existe pas."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"Lecture impossible de {path} : {exc}") from exc
    return validate_references(data)


def run_calibration(camera, path=DEFAULT_CONFIG_PATH):
    """Calibration interactive : Espace valide la face, q/Échap annule.

    Retourne les références enregistrées, ou None si la calibration est
    annulée ou interrompue (rien n'est alors enregistré).
    """
    references = {}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        for step, color in enumerate(COLORS, start=1):
            bgr = _calibrate_color(camera, color, step)
            if bgr is None:
                print("Calibration annulée, rien n'a été enregistré.")
                return None
            references[color] = bgr
            print(f"{COLOR_NAMES_FR[color]:6} : BGR {bgr}")
    finally:
        cv2.destroyWindow(WINDOW_NAME)
        cv2.waitKey(1)

    for a, b in close_pairs(references):
        print(f"Attention : {COLOR_NAMES_FR[a]} et {COLOR_NAMES_FR[b]} "
              "sont très proches, la reconnaissance risque de les confondre.")
    print(f"Calibration enregistrée : {save_calibration(references, path)}")
    return references


def _calibrate_color(camera, color, step):
    """Attend que l'utilisateur valide une face ; retourne sa couleur centrale."""
    instruction = (f"{step}/{len(COLORS)} Centre {COLOR_NAMES_FR[color]} : "
                   "Espace = valider, q = annuler")
    while True:
        try:
            frame = capture_frame(camera)
        except (CaptureError, CameraError) as exc:
            print(f"Erreur de capture : {exc}")
            return None

        detection = detect_face(frame)
        output = draw_detection(frame, detection)
        cv2.putText(output, instruction, (10, output.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, output)
        key = cv2.waitKey(1) & 0xFF

        if key in KEY_CANCEL or cv2.getWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return None
        if key in KEY_VALIDATE:
            stickers = extract_stickers(frame, detection)
            if stickers is None:
                print("Aucune face détectée, recommencer.")
                continue
            return center_reference(stickers)
