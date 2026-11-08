import argparse
import logging
import sys
from functools import partial
import threading

import cv2

from src.calibration import (CalibrationError, load_calibration,
                             run_calibration)
from src.camera import Camera, CameraError, list_cameras
from src.capture import (KEY_QUIT, WINDOW_NAME, CaptureError, capture_frame,
                         describe_frame, run_preview)
from src.color_recognition import draw_colors, format_face, recognize_face
from src.cube import run_scan
from src.detection import detect_face, draw_detection
from src.grid import draw_grid, extract_stickers

INSET_MARGIN = 10
_LOG = logging.getLogger(__name__)


class _FrameSource:
    """Fournit au robot les images lues par la boucle d'aperçu."""

    def __init__(self):
        self._condition = threading.Condition()
        self._frame = None
        self._version = 0
        self._read_version = 0
        self._error = None

    def publish(self, frame):
        with self._condition:
            self._frame = frame.copy()
            self._version += 1
            self._condition.notify_all()

    def read(self):
        with self._condition:
            while self._version <= self._read_version and self._error is None:
                self._condition.wait()
            if self._error is not None:
                raise self._error
            self._read_version = self._version
            return self._frame.copy()

    def fail(self, error):
        with self._condition:
            if self._error is None:
                self._error = error
            self._condition.notify_all()


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


def _preview_window_closed():
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def _run_action_with_preview(camera, action, overlay, first_frame,
                             frame_callback=None, show_window=True,
                             overlay_error_callback=None):
    source = _FrameSource()
    errors = []
    finished = threading.Event()
    overlay_enabled = overlay is not None
    overlay_error_reported = False

    def image_for_preview(frame):
        nonlocal overlay_enabled, overlay_error_reported
        if not overlay_enabled:
            return frame
        try:
            image = overlay(frame)
            if image is None:
                raise ValueError("La détection n'a renvoyé aucune image d'aperçu.")
            return image
        except Exception as exc:
            overlay_enabled = False
            if not overlay_error_reported:
                overlay_error_reported = True
                _LOG.exception("Détection indisponible; le flux brut est conservé.")
                if overlay_error_callback is None:
                    print("Détection indisponible; aperçu brut conservé : %s" % exc)
                else:
                    overlay_error_callback(exc)
            return frame

    def execute_action():
        try:
            action(source)
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=execute_action, name="rubiks-robot-action")
    source.publish(first_frame)
    preview_visible = show_window
    camera_error = None
    worker_started = False
    try:
        image = image_for_preview(first_frame)
        if frame_callback is not None:
            frame_callback(image.copy())
        if show_window:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKey(1) & 0xFF
            if key in KEY_QUIT or _preview_window_closed():
                cv2.destroyWindow(WINDOW_NAME)
                preview_visible = False

        worker.start()
        worker_started = True
        while not finished.is_set():
            try:
                frame = capture_frame(camera)
            except (CaptureError, CameraError) as exc:
                camera_error = exc
                source.fail(exc)
                break
            source.publish(frame)
            image = image_for_preview(frame)
            if frame_callback is not None:
                frame_callback(image.copy())
            if preview_visible:
                cv2.imshow(WINDOW_NAME, image)
                key = cv2.waitKey(1) & 0xFF
                if key in KEY_QUIT or _preview_window_closed():
                    cv2.destroyWindow(WINDOW_NAME)
                    preview_visible = False
    finally:
        if worker_started:
            if worker.is_alive():
                source.fail(CaptureError("L'aperçu s'est arrêté avant la fin des captures."))
            worker.join()
        if preview_visible:
            cv2.destroyWindow(WINDOW_NAME)
        if show_window:
            cv2.waitKey(1)

    if camera_error is not None:
        raise camera_error
    if errors:
        raise errors[0]


def run_on_camera(requested_index, calibrate=False, scan=False,
                  references=None, action=None, available_indexes=None,
                  frame_callback=None, show_window=True, camera_backend=None,
                  overlay_error_callback=None):
    if action is not None and (calibrate or scan):
        raise ValueError("Une action robot ne peut pas être combinée à une calibration ou un scan.")

    detected_here = available_indexes is None
    indexes = list_cameras() if detected_here else available_indexes
    if not indexes:
        print("Aucune caméra détectée.")
        return 1
    if detected_here:
        print(f"Caméras détectées : {indexes}")

    index = indexes[0] if requested_index is None else requested_index
    if index not in indexes:
        print(f"Caméra {index} indisponible. Choisir parmi : {indexes}")
        return 1

    try:
        camera = Camera(index) if camera_backend is None else Camera(index, backend=camera_backend)
        with camera as cam:
            info = cam.info()
            print(f"Caméra {info['index']} ouverte ({info['backend']}) : "
                  f"{info['width']}x{info['height']} @ {info['fps']:.0f} fps")

            first_frame = capture_frame(cam)
            frame_info = describe_frame(first_frame)
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

            if action is not None:
                if show_window:
                    print("Aperçu pendant le mouvement : [q]/[Échap] ferme la fenêtre, "
                          "le robot et les captures continuent.")
                else:
                    print("Aperçu transmis à l'interface graphique.")
                overlay = partial(show_detection, references=references)
                _run_action_with_preview(
                        cam, action, overlay, first_frame, frame_callback, show_window,
                        overlay_error_callback)
                return 0

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
