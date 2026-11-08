"""Assemblage des six faces scannées en chaîne de définition RubikTwoPhase.

Format attendu par RubikTwoPhase (voir enums.py du solveur) : 54 lettres
U, R, F, D, L, B, faces dans l'ordre U, R, F, D, L, B, facettes 1 à 9 ligne
par ligne. Chaque lettre désigne la face dont le centre a la même couleur.

Patron de référence (chaque face est lue dans ce sens) :

            U1 U2 U3
            U4 U5 U6
            U7 U8 U9
 L1 L2 L3   F1 F2 F3   R1 R2 R3   B1 B2 B3
 L4 L5 L6   F4 F5 F6   R4 R5 R6   B4 B5 B6
 L7 L8 L9   F7 F8 F9   R7 R8 R9   B7 B8 B9
            D1 D2 D3
            D4 D5 D6
            D7 D8 D9
"""

from pathlib import Path

import cv2

from .camera import CameraError
from .capture import WINDOW_NAME, CaptureError, capture_frame
from .color_recognition import LABELS, UNKNOWN_LABEL, recognize_face
from .detection import detect_face
from .grid import extract_stickers

FACE_ORDER = "URFDLB"
SCAN_ORDER = "FRBLUD"   # on tourne autour de l'axe vertical, puis haut et bas
CENTER_INDEX = 4
# Face qui doit se trouver en haut de l'image quand on scanne chaque face,
# pour que la lecture corresponde au patron ci-dessus.
TOP_NEIGHBOR = {"U": "B", "R": "U", "F": "U", "D": "F", "L": "U", "B": "U"}
# Orientation de référence : blanc en haut, vert devant.
DEFAULT_SCHEME = {"U": "white", "R": "red", "F": "green",
                  "D": "yellow", "L": "orange", "B": "blue"}
COLOR_NAMES_FR = {
    "white": "BLANC", "yellow": "JAUNE", "red": "ROUGE",
    "orange": "ORANGE", "green": "VERT", "blue": "BLEU",
}
DEFAULT_OUTPUT_PATH = Path("cubestring.txt")
KEY_VALIDATE = {ord(" "), ord("c")}
KEY_BACK = {ord("r"), 8}       # r ou Retour arrière : refaire la face précédente
KEY_CANCEL = {ord("q"), 27}


class CubeError(ValueError):
    """Les faces scannées ne permettent pas de construire la chaîne."""


def scan_instruction(face, scheme=DEFAULT_SCHEME):
    """Consigne d'orientation pour scanner une face (sans accents, pour OpenCV)."""
    front = COLOR_NAMES_FR[scheme[face]]
    top = COLOR_NAMES_FR[scheme[TOP_NEIGHBOR[face]]]
    return f"Face {face} : centre {front} vers la camera, {top} en haut"


def check_face(colors, faces):
    """Vérifie qu'une face scannée peut être ajoutée ; lève CubeError sinon."""
    if len(colors) != 9:
        raise CubeError(f"9 couleurs attendues, {len(colors)} reçues.")
    unknown = [i + 1 for i, color in enumerate(colors) if color is None]
    if unknown:
        raise CubeError(f"Couleur non reconnue pour les stickers {unknown}.")
    center = colors[CENTER_INDEX]
    for other, other_colors in faces.items():
        if other_colors[CENTER_INDEX] == center:
            raise CubeError(f"Centre {center} déjà scanné sur la face {other}.")


def to_cubestring(faces):
    """Chaîne de 54 lettres à partir de {face: 9 couleurs}."""
    missing = [face for face in FACE_ORDER if face not in faces]
    if missing:
        raise CubeError(f"Faces manquantes : {', '.join(missing)}.")

    center_to_face = {}
    for face in FACE_ORDER:
        center = faces[face][CENTER_INDEX]
        if center is None or center in center_to_face:
            raise CubeError(f"Centre de la face {face} invalide ou en double.")
        center_to_face[center] = face

    letters = []
    for face in FACE_ORDER:
        for index, color in enumerate(faces[face]):
            if color not in center_to_face:
                raise CubeError(f"Sticker {face}{index + 1} : couleur {color!r} "
                                "ne correspond à aucun centre.")
            letters.append(center_to_face[color])
    cubestring = "".join(letters)
    counts = {face: cubestring.count(face) for face in FACE_ORDER}
    invalid = ["{}={}".format(face, count) for face, count in counts.items() if count != 9]
    if invalid:
        raise CubeError("Répartition des couleurs invalide (9 attendues par face) : "
                        + ", ".join(invalid))
    return cubestring


def format_net(faces):
    """Patron texte du cube (couleurs W, Y, R, O, G, B, ? ; '.' si non scannée)."""
    def rows(face):
        colors = faces.get(face)
        if colors is None:
            return [". . ."] * 3
        labels = [LABELS.get(c, UNKNOWN_LABEL) for c in colors]
        return [" ".join(labels[r * 3:r * 3 + 3]) for r in range(3)]

    pad = " " * 8
    lines = [pad + row for row in rows("U")]
    middle = [rows(face) for face in "LFRB"]
    lines += ["   ".join(parts) for parts in zip(*middle)]
    lines += [pad + row for row in rows("D")]
    return "\n".join(lines)


def save_cubestring(cubestring, path=DEFAULT_OUTPUT_PATH):
    """Enregistre la chaîne (une ligne) et retourne le chemin."""
    if len(cubestring) != 54 or set(cubestring) - set(FACE_ORDER):
        raise CubeError(f"Chaîne invalide : {cubestring!r}")
    path = Path(path)
    path.write_text(cubestring + "\n", encoding="utf-8")
    return path


def run_scan(camera, references=None, overlay=None, path=DEFAULT_OUTPUT_PATH,
             scheme=DEFAULT_SCHEME):
    """Scan guidé des six faces ; retourne la chaîne enregistrée ou None.

    Espace valide la face, r revient à la face précédente, q annule.
    `overlay(frame)` fournit l'image affichée (détection, couleurs...).
    """
    faces = {}
    step = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    try:
        while step < len(SCAN_ORDER):
            face = SCAN_ORDER[step]
            action, colors = _wait_face(camera, face, step, references,
                                        overlay, scheme)
            if action == "cancel":
                print("Scan annulé.")
                return None
            if action == "back":
                if step > 0:
                    step -= 1
                    faces.pop(SCAN_ORDER[step], None)
                    print(f"Retour à la face {SCAN_ORDER[step]}.")
                continue
            others = {f: c for f, c in faces.items() if f != face}
            try:
                check_face(colors, others)
            except CubeError as exc:
                print(f"Face {face} refusée : {exc}")
                continue
            faces[face] = colors
            print(f"Face {face} enregistrée.")
            step += 1
    finally:
        cv2.destroyWindow(WINDOW_NAME)
        cv2.waitKey(1)

    print(format_net(faces))
    try:
        cubestring = to_cubestring(faces)
    except CubeError as exc:
        print(f"Impossible de construire la chaîne : {exc}")
        return None
    print(f"Chaîne : {cubestring}")
    print(f"Enregistrée dans {save_cubestring(cubestring, path)}")
    return cubestring


def _wait_face(camera, face, step, references, overlay, scheme):
    """Affiche le flux jusqu'à une action ; retourne (action, couleurs)."""
    progress = f"{step + 1}/{len(SCAN_ORDER)} {scan_instruction(face, scheme)}"
    keys_help = "Espace = valider   r = face precedente   q = annuler"
    while True:
        try:
            frame = capture_frame(camera)
        except (CaptureError, CameraError) as exc:
            print(f"Erreur de capture : {exc}")
            return "cancel", None

        output = overlay(frame) if overlay else frame.copy()
        height = output.shape[0]
        for i, text in enumerate((progress, keys_help)):
            cv2.putText(output, text, (10, height - 40 + 25 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, output)
        key = cv2.waitKey(1) & 0xFF

        if key in KEY_CANCEL or cv2.getWindowProperty(
                WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            return "cancel", None
        if key in KEY_BACK:
            return "back", None
        if key in KEY_VALIDATE:
            stickers = extract_stickers(frame, detect_face(frame))
            if stickers is None:
                print("Aucune face détectée, recommencer.")
                continue
            return "validate", recognize_face(stickers, references)
