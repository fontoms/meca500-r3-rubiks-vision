"""Extraction des neuf stickers d'une face détectée.

La face est redressée en vue frontale carrée, puis découpée en grille 3×3.
Les stickers sont ordonnés ligne par ligne, de haut-gauche à bas-droit
(indices 0 à 8), ce qui correspond à la numérotation des facettes 1 à 9
attendue par RubikTwoPhase pour chaque face.
"""

from dataclasses import dataclass

import cv2
import numpy as np

GRID_SIZE = 3
DEFAULT_FACE_SIZE = 300   # côté de la face redressée, en pixels
DEFAULT_PATCH_RATIO = 0.5  # part centrale de chaque case conservée


@dataclass
class FaceStickers:
    face: np.ndarray      # vue frontale carrée de la face
    cells: list           # 9 cases complètes, ordre ligne par ligne
    patches: list         # 9 zones centrales (sans les bords noirs), même ordre


def warp_face(frame, corners, size=DEFAULT_FACE_SIZE):
    """Redresse la face délimitée par `corners` (HG, HD, BD, BG) en carré."""
    if frame is None or frame.size == 0:
        raise ValueError("Image vide.")
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if cv2.contourArea(corners) <= 0:
        raise ValueError("Coins dégénérés.")
    target = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    matrix = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(frame, matrix, (size, size))


def split_cells(face):
    """Découpe une face carrée en 9 cases, ligne par ligne."""
    height, width = face.shape[:2]
    ys = np.linspace(0, height, GRID_SIZE + 1).round().astype(int)
    xs = np.linspace(0, width, GRID_SIZE + 1).round().astype(int)
    return [face[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            for r in range(GRID_SIZE) for c in range(GRID_SIZE)]


def center_patch(cell, ratio=DEFAULT_PATCH_RATIO):
    """Zone centrale d'une case, à l'écart des bords noirs entre stickers."""
    if not 0 < ratio <= 1:
        raise ValueError(f"Ratio invalide : {ratio}")
    height, width = cell.shape[:2]
    dy = round(height * (1 - ratio) / 2)
    dx = round(width * (1 - ratio) / 2)
    return cell[dy:height - dy, dx:width - dx]


def extract_stickers(frame, detection, size=DEFAULT_FACE_SIZE,
                     patch_ratio=DEFAULT_PATCH_RATIO):
    """Retourne les FaceStickers d'une face détectée, ou None sans détection."""
    if detection is None:
        return None
    face = warp_face(frame, detection.corners, size)
    cells = split_cells(face)
    patches = [center_patch(cell, patch_ratio) for cell in cells]
    return FaceStickers(face, cells, patches)


def draw_grid(stickers, patch_ratio=DEFAULT_PATCH_RATIO):
    """Copie de la face redressée avec la grille, les zones centrales et les numéros."""
    output = stickers.face.copy()
    size = output.shape[0]
    step = size / GRID_SIZE
    for i in range(1, GRID_SIZE):
        offset = round(i * step)
        cv2.line(output, (offset, 0), (offset, size), (0, 255, 0), 1)
        cv2.line(output, (0, offset), (size, offset), (0, 255, 0), 1)

    half = step * patch_ratio / 2
    for index in range(GRID_SIZE * GRID_SIZE):
        row, col = divmod(index, GRID_SIZE)
        cx, cy = (col + 0.5) * step, (row + 0.5) * step
        cv2.rectangle(output, (round(cx - half), round(cy - half)),
                      (round(cx + half), round(cy + half)), (255, 255, 0), 1)
        cv2.putText(output, str(index + 1), (round(col * step) + 4, round(row * step) + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return output
