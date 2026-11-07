import cv2
import numpy as np
import pytest

from src.detection import detect_face
from src.grid import FACE_SIZE, cell_bounds, draw_cells, split_cells, warp_face
from tests.test_detection import STICKER_COLORS, make_face, warp

COLOR_TOLERANCE = 20


def cell_colors(cells):
    return [tuple(cell.reshape(-1, 3).mean(axis=0)) for cell in cells]


def assert_colors_match(cells):
    assert len(cells) == 9
    for i, (color, expected) in enumerate(zip(cell_colors(cells), STICKER_COLORS)):
        assert np.abs(np.array(color) - expected).max() < COLOR_TOLERANCE, \
            f"case {i} : {color} au lieu de {expected}"


def rectified_cells(image):
    detection = detect_face(image)
    assert detection is not None, "aucune face détectée"
    return split_cells(warp_face(image, detection.corners))


def test_warp_face_is_square():
    image, _ = make_face()
    face = warp_face(image, detect_face(image).corners)
    assert face.shape == (FACE_SIZE, FACE_SIZE, 3)


def test_frontal_face_cells_in_reading_order():
    image, _ = make_face()
    assert_colors_match(rectified_cells(image))


@pytest.mark.parametrize("angle", [12, -20, 30])
def test_rotated_face(angle):
    image, corners = make_face()
    matrix = cv2.getRotationMatrix2D((320, 240), angle, 1.0)
    assert_colors_match(rectified_cells(warp(image, corners, matrix)[0]))


def test_face_in_perspective():
    image, corners = make_face()
    src = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
    dst = np.float32([[60, 30], [600, 0], [640, 480], [0, 440]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    assert_colors_match(rectified_cells(warp(image, corners, matrix, perspective=True)[0]))


def test_cell_bounds_are_disjoint_and_inside_face():
    bounds = cell_bounds()
    assert len(bounds) == 9
    mask = np.zeros((FACE_SIZE, FACE_SIZE), np.int32)
    for x0, y0, x1, y1 in bounds:
        assert 0 <= x0 < x1 <= FACE_SIZE and 0 <= y0 < y1 <= FACE_SIZE
        mask[y0:y1, x0:x1] += 1
    assert mask.max() == 1


def test_split_cells_have_equal_size():
    cells = split_cells(np.zeros((300, 300, 3), np.uint8))
    assert len({cell.shape for cell in cells}) == 1


@pytest.mark.parametrize("face", [None, np.zeros((300, 200, 3), np.uint8)])
def test_split_rejects_non_square(face):
    with pytest.raises(ValueError):
        split_cells(face)


def test_draw_cells_does_not_modify_input():
    face = np.zeros((300, 300, 3), np.uint8)
    out = draw_cells(face)
    assert not face.any() and out.any()
