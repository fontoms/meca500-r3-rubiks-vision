import argparse
import sys
from functools import partial

import cv2

from src.calibration import (CalibrationError, load_calibration,
                             run_calibration)
from src.camera import Camera, CameraError, list_cameras
from src.capture import CaptureError, capture_frame, describe_frame, run_preview
from src.color_recognition import draw_colors, format_face, recognize_face
from src.cube import run_scan
from src.detection import detect_face, draw_detection
from src.grid import draw_grid, extract_stickers

INSET_MARGIN = 10


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Rubik's Cube Vision")
    parser.add_argument("--camera", type=int, default=None, metavar="INDEX",
                        help="index de la caméra à utiliser "
                             "(par défaut : la première détectée)")
    parser.add_argument("--image", default=None, metavar="FICHIER",
                        help="détecter la face sur une image enregistrée "
                             "au lieu de la webcam")
    parser.add_argument("--calibrate", action="store_true",
                        help="calibrer les couleurs avec la webcam "
                             "(enregistrées dans config/colors.json)")
    parser.add_argument("--scan", action="store_true",
                        help="scanner les six faces et produire la chaîne "
                             "RubikTwoPhase (cubestring.txt)")
    return parser.parse_args(argv)


def show_detection(frame, references=None):
    detection = detect_face(frame)
    output = draw_detection(frame, detection)
    stickers = extract_stickers(frame, detection)
    if stickers is not None:
        paste_inset(output, draw_face(stickers, references))
    return output


def draw_face(stickers, references=None):
    colors = recognize_face(stickers, references)
    return draw_colors(draw_grid(stickers), colors)


def paste_inset(image, inset):
    """Incruste `inset` en haut à droite de `image` s'il y a la place."""
    h, w = inset.shape[:2]
    x = image.shape[1] - w - INSET_MARGIN
    if x < 0 or h + INSET_MARGIN > image.shape[0]:
        return
    image[INSET_MARGIN:INSET_MARGIN + h, x:x + w] = inset


def run_on_image(path, references=None):
    frame = cv2.imread(path)
    if frame is None:
        print(f"Impossible de lire l'image : {path}")
        return 1
    detection = detect_face(frame)
    if detection is None:
        print("Aucune face détectée.")
    else:
        print(f"Face détectée ({len(detection.stickers)} stickers), coins :")
        for label, (x, y) in zip(("haut-gauche", "haut-droit", "bas-droit",
                                  "bas-gauche"), detection.corners):
            print(f"  {label:12} ({x:.0f}, {y:.0f})")
    cv2.imshow("Detection", draw_detection(frame, detection))
    stickers = extract_stickers(frame, detection)
    if stickers is not None:
        print("Couleurs reconnues :")
        print(format_face(recognize_face(stickers, references)))
        cv2.imshow("Face redressee", draw_face(stickers, references))
    print("Appuyer sur une touche pour fermer.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def run_on_camera(requested_index, calibrate=False, scan=False,
                  references=None):
    indexes = list_cameras()
    if not indexes:
        print("Aucune caméra détectée.")
        return 1
    print(f"Caméras détectées : {indexes}")

    index = indexes[0] if requested_index is None else requested_index
    if index not in indexes:
        print(f"Caméra {index} indisponible. Choisir parmi : {indexes}")
        return 1

    try:
        with Camera(index) as cam:
            info = cam.info()
            print(f"Caméra {info['index']} ouverte ({info['backend']}) : "
                  f"{info['width']}x{info['height']} @ {info['fps']:.0f} fps")

            frame_info = describe_frame(capture_frame(cam))
            print(f"Image : {frame_info['width']}x{frame_info['height']}, "
                  f"{frame_info['channels']} canaux {frame_info['format']}, "
                  f"{frame_info['dtype']}")

            if calibrate:
                print("Calibration : présenter la face dont le centre a la "
                      "couleur demandée, puis [Espace] ; [q] annule.")
                return 0 if run_calibration(cam) else 1

            if scan:
                print("Scan : suivre les consignes à l'écran. [Espace] valide, "
                      "[r] refait la face précédente, [q] annule.")
                overlay = partial(show_detection, references=references)
                return 0 if run_scan(cam, references, overlay) else 1

            print("Aperçu : [c]/[Espace] capturer, [q]/[Échap] quitter.")
            saved = run_preview(
                cam, overlay=partial(show_detection, references=references))
    except (CameraError, CaptureError) as exc:
        print(f"Erreur : {exc}")
        return 1

    print(f"{len(saved)} image(s) enregistrée(s). Caméra libérée.")
    return 0


def load_references():
    """Références calibrées, ou None (seuils HSV par défaut)."""
    try:
        references = load_calibration()
    except CalibrationError as exc:
        print(f"Calibration ignorée : {exc}")
        return None
    if references is None:
        print("Pas de calibration : seuils de couleur par défaut "
              "(lancer --calibrate pour calibrer).")
    else:
        print("Calibration chargée.")
    return references


def main(argv=None):
    args = parse_args(argv)
    if args.calibrate:
        return run_on_camera(args.camera, calibrate=True)
    references = load_references()
    if args.image:
        return run_on_image(args.image, references)
    return run_on_camera(args.camera, scan=args.scan, references=references)


if __name__ == "__main__":
    sys.exit(main())
