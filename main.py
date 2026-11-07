import argparse
import sys

import cv2

from src.camera import Camera, CameraError, list_cameras
from src.capture import CaptureError, capture_frame, describe_frame, run_preview
from src.detection import detect_face, draw_detection


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Rubik's Cube Vision")
    parser.add_argument("--camera", type=int, default=None, metavar="INDEX",
                        help="index de la caméra à utiliser "
                             "(par défaut : la première détectée)")
    parser.add_argument("--image", default=None, metavar="FICHIER",
                        help="détecter la face sur une image enregistrée "
                             "au lieu de la webcam")
    parser.add_argument("--exposure", type=float, default=None, metavar="VALEUR",
                        help="désactive l'exposition automatique et fixe cette "
                             "valeur (souvent négative, ex. -5)")
    parser.add_argument("--lock-wb", action="store_true",
                        help="désactive la balance des blancs automatique")
    return parser.parse_args(argv)


def show_detection(frame):
    return draw_detection(frame, detect_face(frame))


def run_on_image(path):
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
    print("Appuyer sur une touche pour fermer.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def run_on_camera(requested_index, exposure=None, lock_white_balance=False):
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
        with Camera(index, exposure=exposure,
                    lock_white_balance=lock_white_balance) as cam:
            info = cam.info()
            print(f"Caméra {info['index']} ouverte ({info['backend']}) : "
                  f"{info['width']}x{info['height']} @ {info['fps']:.0f} fps")

            frame_info = describe_frame(capture_frame(cam))
            print(f"Image : {frame_info['width']}x{frame_info['height']}, "
                  f"{frame_info['channels']} canaux {frame_info['format']}, "
                  f"{frame_info['dtype']}")

            print("Aperçu : [c]/[Espace] capturer, [q]/[Échap] quitter.")
            saved = run_preview(cam, overlay=show_detection)
    except (CameraError, CaptureError) as exc:
        print(f"Erreur : {exc}")
        return 1

    print(f"{len(saved)} image(s) enregistrée(s). Caméra libérée.")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.image:
        return run_on_image(args.image)
    return run_on_camera(args.camera, args.exposure, args.lock_wb)


if __name__ == "__main__":
    sys.exit(main())
