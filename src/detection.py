"""Localisation d'une face du Rubik's Cube dans une image.

Les stickers colorés contrastent nettement avec le plastique noir du cube :
on détecte ces petits carrés, on garde le plus grand groupe de carrés voisins
de taille similaire, puis on en déduit les quatre coins de la face.
"""

from dataclasses import dataclass

import cv2
import numpy as np

MIN_STICKERS = 5
MIN_STICKER_AREA_RATIO = 0.0005  # par rapport à l'aire de l'image
MAX_STICKER_AREA_RATIO = 0.05
MIN_ASPECT_RATIO = 0.6           # petit côté / grand côté
MIN_FILL_RATIO = 0.75            # aire du contour / aire du rectangle englobant
NEIGHBOR_DISTANCE = 1.8          # en nombre de côtés de sticker (diagonale incluse)
MIN_QUAD_COVERAGE = 0.9


@dataclass
class FaceDetection:
    corners: np.ndarray   # (4, 2) float32 : haut-gauche, haut-droit, bas-droit, bas-gauche
    stickers: list        # quadrilatères (4, 2) des stickers retenus


def detect_face(frame):
    """Retourne une FaceDetection, ou None si aucune face n'est trouvée."""
    if frame is None or frame.size == 0:
        return None
    stickers = find_sticker_candidates(frame)
    group = largest_sticker_group(stickers)
    if len(group) < MIN_STICKERS:
        return None
    return FaceDetection(face_corners(group), group)


def find_sticker_candidates(frame):
    """Retourne les quadrilatères ressemblant à des stickers."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_area = gray.shape[0] * gray.shape[1]
    min_area = MIN_STICKER_AREA_RATIO * image_area
    max_area = MAX_STICKER_AREA_RATIO * image_area
    candidates = [quad for quad in map(_as_square_quad, contours)
                  if quad is not None and min_area <= cv2.contourArea(quad) <= max_area]
    return _remove_duplicates(candidates)


def _as_square_quad(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    (_, _), (w, h), _ = cv2.minAreaRect(approx)
    if min(w, h) == 0 or min(w, h) / max(w, h) < MIN_ASPECT_RATIO:
        return None
    if cv2.contourArea(contour) / (w * h) < MIN_FILL_RATIO:
        return None
    return approx.reshape(4, 2).astype(np.float32)


def _remove_duplicates(quads):
    """Un même sticker peut produire plusieurs contours imbriqués : on garde le plus grand."""
    kept = []
    for quad in sorted(quads, key=cv2.contourArea, reverse=True):
        center, side = quad.mean(axis=0), _side(quad)
        if all(np.linalg.norm(center - k.mean(axis=0)) > 0.5 * side for k in kept):
            kept.append(quad)
    return kept


def largest_sticker_group(stickers):
    """Plus grand ensemble de stickers voisins et de taille similaire."""
    centers = [s.mean(axis=0) for s in stickers]
    sides = [_side(s) for s in stickers]

    def linked(i, j):
        ratio = sides[i] / sides[j]
        distance = np.linalg.norm(centers[i] - centers[j])
        return 0.7 <= ratio <= 1.4 and distance < NEIGHBOR_DISTANCE * (sides[i] + sides[j]) / 2

    best, unvisited = [], set(range(len(stickers)))
    while unvisited:
        stack, component = [unvisited.pop()], []
        while stack:
            i = stack.pop()
            component.append(i)
            neighbors = {j for j in unvisited if linked(i, j)}
            unvisited -= neighbors
            stack.extend(neighbors)
        if len(component) > len(best):
            best = component
    return [stickers[i] for i in best]


def face_corners(stickers):
    """Quatre coins ordonnés de la zone couverte par les stickers."""
    points = np.vstack(stickers).astype(np.float32)
    hull = cv2.convexHull(points)
    box = cv2.boxPoints(cv2.minAreaRect(hull))

    # L'enveloppe suit la perspective, mais si un sticker de coin manque elle
    # est tronquée : on se rabat alors sur le rectangle englobant minimal.
    quad = _hull_to_quad(hull)
    if quad is not None and cv2.contourArea(quad) >= MIN_QUAD_COVERAGE * cv2.contourArea(box):
        return order_corners(quad)
    return order_corners(box)


def _hull_to_quad(hull):
    perimeter = cv2.arcLength(hull, True)
    for factor in (0.02, 0.04, 0.06, 0.08, 0.1):
        approx = cv2.approxPolyDP(hull, factor * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return None


def order_corners(points):
    """Ordonne 4 points : haut-gauche, haut-droit, bas-droit, bas-gauche."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    clockwise = points[np.argsort(angles)]  # sens horaire (axe y vers le bas)
    start = np.argmin(clockwise.sum(axis=1))
    return np.roll(clockwise, -start, axis=0)


def draw_detection(frame, detection):
    """Retourne une copie de l'image avec la face détectée dessinée."""
    output = frame.copy()
    if detection is None:
        cv2.putText(output, "Aucune face detectee", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return output
    for sticker in detection.stickers:
        cv2.polylines(output, [sticker.astype(np.int32)], True, (255, 255, 0), 1)
    cv2.polylines(output, [detection.corners.astype(np.int32)], True, (0, 255, 0), 2)
    for label, (x, y) in zip(("HG", "HD", "BD", "BG"), detection.corners.astype(int)):
        cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(output, label, (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return output


def _side(quad):
    return float(np.sqrt(cv2.contourArea(quad)))
