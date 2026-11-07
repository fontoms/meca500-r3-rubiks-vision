import cv2
import numpy as np
import pytest

from src.detection import detect_face, draw_detection, order_corners

STICKER_COLORS = [
    (255, 255, 255), (0, 213, 255), (0, 0, 200),
    (0, 100, 255), (60, 160, 0), (180, 70, 0),
    (0, 213, 255), (255, 255, 255), (0, 0, 200),
]
TOLERANCE = 8  # pixels


def make_face(width=640, height=480, side=210, background=(140, 130, 120),
              skip=None, colors=STICKER_COLORS):
    """Face de cube centrée et alignée ; retourne (image, coins des stickers)."""
    image = np.full((height, width, 3), background, np.uint8)
    x0, y0 = (width - side) // 2, (height - side) // 2
    cv2.rectangle(image, (x0, y0), (x0 + side, y0 + side), (20, 20, 20), -1)

    margin, gap = side * 0.04, side * 0.04
    cell = (side - 2 * margin - 2 * gap) / 3
    for i, color in enumerate(colors):
        if i == skip:
            continue
        row, col = divmod(i, 3)
        x = x0 + margin + col * (cell + gap)
        y = y0 + margin + row * (cell + gap)
        cv2.rectangle(image, (round(x), round(y)),
                      (round(x + cell), round(y + cell)), color, -1)

    a, b = x0 + margin, x0 + side - margin
    c, d = y0 + margin, y0 + side - margin
    corners = np.float32([[a, c], [b, c], [b, d], [a, d]])
    return image, corners


def warp(image, corners, matrix, perspective=False):
    h, w = image.shape[:2]
    border = image[0, 0].tolist()
    if perspective:
        out = cv2.warpPerspective(image, matrix, (w, h), borderValue=border)
        pts = cv2.perspectiveTransform(corners[None], matrix)[0]
    else:
        out = cv2.warpAffine(image, matrix, (w, h), borderValue=border)
        pts = cv2.transform(corners[None], matrix)[0]
    return out, pts


def assert_corners(detection, expected):
    assert detection is not None, "aucune face détectée"
    error = np.linalg.norm(detection.corners - order_corners(expected), axis=1)
    assert error.max() < TOLERANCE, f"erreur des coins : {error}"


def test_frontal_face():
    image, corners = make_face()
    assert_corners(detect_face(image), corners)


def test_detects_stickers():
    image, _ = make_face()
    assert len(detect_face(image).stickers) == 9


def test_dark_blue_stickers_on_dark_plastic():
    # bleu sombre : presque le même niveau de gris que le plastique, mais saturé
    image, corners = make_face(colors=[(120, 30, 0)] * 9)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert abs(int(gray[240, 320]) - 20) < 15  # contraste en gris très faible
    assert_corners(detect_face(image), corners)


@pytest.mark.parametrize("angle", [10, 30, -25, 45])
def test_rotated_face(angle):
    image, corners = make_face()
    matrix = cv2.getRotationMatrix2D((320, 240), angle, 1.0)
    assert_corners(detect_face(warp(image, corners, matrix)[0]),
                   warp(image, corners, matrix)[1])


@pytest.mark.parametrize("dx, dy, scale", [(-150, -80, 0.7), (140, 90, 0.6), (0, 0, 1.4)])
def test_shifted_and_scaled_face(dx, dy, scale):
    image, corners = make_face()
    matrix = cv2.getRotationMatrix2D((320, 240), 0, scale)
    matrix[:, 2] += (dx, dy)
    warped, expected = warp(image, corners, matrix)
    assert_corners(detect_face(warped), expected)


def test_face_in_perspective():
    image, corners = make_face()
    src = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
    dst = np.float32([[60, 30], [600, 0], [640, 480], [0, 440]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped, expected = warp(image, corners, matrix, perspective=True)
    assert_corners(detect_face(warped), expected)


@pytest.mark.parametrize("skip", [0, 4, 8])
def test_missing_sticker(skip):
    image, corners = make_face(skip=skip)
    assert_corners(detect_face(image), corners)


def test_noisy_image():
    image, corners = make_face()
    noise = np.random.default_rng(0).normal(0, 12, image.shape)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    assert_corners(detect_face(noisy), corners)


def test_dark_background():
    image, corners = make_face(background=(15, 15, 15))
    assert_corners(detect_face(image), corners)


def test_ignores_distractor_squares():
    image, corners = make_face()
    cv2.rectangle(image, (20, 20), (110, 110), (0, 0, 0), -1)
    cv2.rectangle(image, (560, 380), (590, 410), (255, 0, 255), -1)
    assert_corners(detect_face(image), corners)


@pytest.mark.parametrize("image", [
    np.full((480, 640, 3), 128, np.uint8),
    np.random.default_rng(1).integers(0, 256, (480, 640, 3), np.uint8),
    None,
])
def test_no_face(image):
    assert detect_face(image) is None


def test_grayscale_input():
    image, corners = make_face()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    assert_corners(detect_face(gray), corners)


@pytest.mark.parametrize("points", [
    [[10, 10], [100, 10], [100, 100], [10, 100]],
    [[100, 100], [10, 10], [10, 100], [100, 10]],
])
def test_order_corners(points):
    expected = [[10, 10], [100, 10], [100, 100], [10, 100]]
    assert np.array_equal(order_corners(points), np.float32(expected))


def test_draw_detection_does_not_modify_input():
    image, _ = make_face()
    original = image.copy()
    drawn = draw_detection(image, detect_face(image))
    assert np.array_equal(image, original)
    assert not np.array_equal(drawn, image)


def test_draw_without_detection():
    image = np.zeros((100, 100, 3), np.uint8)
    assert draw_detection(image, None).any()
