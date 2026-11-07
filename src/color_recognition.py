"""Reconnaissance de la couleur des stickers.

Chaque zone centrale est réduite à sa couleur médiane (robuste au bruit et
aux reflets). Une zone trop sombre est inconnue. Sinon :
- sans calibration, la couleur est classée par seuils HSV fixes ;
- avec calibration, c'est la couleur de référence la plus proche dans
  l'espace Lab (plus fidèle à la perception que HSV pour une distance).
"""

import cv2
import numpy as np

WHITE, YELLOW, RED, ORANGE, GREEN, BLUE = (
    "white", "yellow", "red", "orange", "green", "blue")
COLORS = (WHITE, YELLOW, RED, ORANGE, GREEN, BLUE)

# Seuils par défaut (teinte OpenCV : 0-179, saturation et valeur : 0-255).
MIN_VALUE = 50         # en dessous : plastique noir ou ombre, couleur inconnue
MAX_WHITE_SATURATION = 80
HUE_RANGES = (         # (teinte min incluse, teinte max exclue, couleur)
    (0, 6, RED),
    (6, 20, ORANGE),
    (20, 40, YELLOW),
    (40, 88, GREEN),
    (88, 145, BLUE),
    (145, 180, RED),
)

# Distance Lab : la luminosité (L) compte moins que la teinte (a, b), car elle
# varie surtout avec l'éclairage.
LIGHTNESS_WEIGHT = 0.5
MAX_REFERENCE_DISTANCE = 60  # au-delà : trop loin de toute référence, inconnue

# Couleurs BGR d'affichage et abréviations.
DISPLAY_BGR = {
    WHITE: (255, 255, 255), YELLOW: (0, 220, 255), RED: (0, 0, 220),
    ORANGE: (0, 128, 255), GREEN: (0, 180, 0), BLUE: (220, 80, 0),
}
LABELS = {WHITE: "W", YELLOW: "Y", RED: "R", ORANGE: "O", GREEN: "G", BLUE: "B"}
UNKNOWN_LABEL = "?"


def median_bgr(patch):
    """Couleur BGR médiane d'une zone d'image couleur."""
    if not isinstance(patch, np.ndarray) or patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError("Une zone BGR (hauteur, largeur, 3) est attendue.")
    if patch.size == 0:
        raise ValueError("Zone vide.")
    return np.median(patch.reshape(-1, 3), axis=0)


def bgr_to_hsv(bgr):
    """Convertit une couleur BGR en (teinte, saturation, valeur) OpenCV."""
    pixel = np.uint8([[np.clip(np.round(bgr), 0, 255)]])
    return tuple(int(v) for v in cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0])


def classify_hsv(hsv):
    """Retourne le nom de la couleur, ou None si elle est trop sombre."""
    hue, saturation, value = hsv
    if value < MIN_VALUE:
        return None
    if saturation < MAX_WHITE_SATURATION:
        return WHITE
    for low, high, color in HUE_RANGES:
        if low <= hue < high:
            return color
    return None


def bgr_to_lab(bgr):
    """Convertit une couleur BGR en Lab OpenCV (composantes 0-255)."""
    pixel = np.uint8([[np.clip(np.round(bgr), 0, 255)]])
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0].astype(float)


def lab_distance(bgr_a, bgr_b):
    """Distance perceptuelle entre deux couleurs BGR."""
    diff = bgr_to_lab(bgr_a) - bgr_to_lab(bgr_b)
    diff[0] *= LIGHTNESS_WEIGHT
    return float(np.linalg.norm(diff))


def classify_with_references(bgr, references):
    """Couleur de référence la plus proche, ou None si toutes sont trop loin."""
    if bgr_to_hsv(bgr)[2] < MIN_VALUE:
        return None
    distance, color = min((lab_distance(bgr, ref), color)
                          for color, ref in references.items())
    return color if distance <= MAX_REFERENCE_DISTANCE else None


def classify_patch(patch, references=None):
    """Couleur d'une zone de sticker, ou None si elle est indéterminée.

    `references` ({couleur: BGR}) active la classification calibrée.
    """
    bgr = median_bgr(patch)
    if references:
        return classify_with_references(bgr, references)
    return classify_hsv(bgr_to_hsv(bgr))


def recognize_face(stickers, references=None):
    """Liste des 9 couleurs d'une face (ordre ligne par ligne)."""
    return [classify_patch(patch, references) for patch in stickers.patches]


def draw_colors(face_image, colors):
    """Copie de la face avec, dans chaque case, une pastille de la couleur reconnue."""
    output = face_image.copy()
    step = output.shape[0] / 3
    radius = max(4, round(step * 0.12))
    for index, color in enumerate(colors):
        row, col = divmod(index, 3)
        center = (round((col + 0.75) * step), round((row + 0.75) * step))
        if color is None:
            cv2.circle(output, center, radius, (128, 128, 128), 2)
        else:
            cv2.circle(output, center, radius, DISPLAY_BGR[color], -1)
            cv2.circle(output, center, radius, (0, 0, 0), 1)
        label = LABELS.get(color, UNKNOWN_LABEL)
        cv2.putText(output, label, (center[0] - radius, center[1] - radius - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return output


def format_face(colors):
    """Représentation texte 3×3 des couleurs (W, Y, R, O, G, B ou ?)."""
    labels = [LABELS.get(color, UNKNOWN_LABEL) for color in colors]
    return "\n".join(" ".join(labels[row * 3:row * 3 + 3]) for row in range(3))
