"""Redressement d'une face détectée et découpage en neuf cases."""

import cv2
import numpy as np

FACE_SIZE = 300       # côté de la vue de face, en pixels
CELL_MARGIN = 0.25    # part du côté de chaque case ignorée de chaque côté


def warp_face(frame, corners, size=FACE_SIZE):
    """Retourne la face vue de face, en image carrée size x size.

    `corners` : haut-gauche, haut-droit, bas-droit, bas-gauche (voir
    `detection.order_corners`).
    """
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    target = np.float32([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]])
    matrix = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(frame, matrix, (size, size))


def cell_bounds(size=FACE_SIZE, margin=CELL_MARGIN):
    """Rectangles (x0, y0, x1, y1) des 9 cases, ligne par ligne.

    Chaque rectangle est réduit au centre de la case pour éviter les bords
    du sticker et le plastique noir.
    """
    step = size / 3
    inset = step * margin
    bounds = []
    for row in range(3):
        for col in range(3):
            bounds.append((round(col * step + inset), round(row * step + inset),
                           round((col + 1) * step - inset), round((row + 1) * step - inset)))
    return bounds


def split_cells(face, margin=CELL_MARGIN):
    """Découpe une face carrée en 9 images, de haut-gauche à bas-droite."""
    if face is None or face.ndim < 2 or face.shape[0] != face.shape[1]:
        raise ValueError("La face doit être une image carrée.")
    return [face[y0:y1, x0:x1] for x0, y0, x1, y1 in cell_bounds(face.shape[0], margin)]


def draw_cells(face, margin=CELL_MARGIN):
    """Retourne une copie de la face avec les 9 cases d'échantillonnage dessinées."""
    output = face.copy()
    for i, (x0, y0, x1, y1) in enumerate(cell_bounds(face.shape[0], margin)):
        cv2.rectangle(output, (x0, y0), (x1, y1), (255, 0, 255), 2)
        cv2.putText(output, str(i), (x0 + 4, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    return output
