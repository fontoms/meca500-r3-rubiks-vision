import cv2
import numpy as np
import pytest

from src.camera import Camera, list_cameras
from src import capture as capture_module
from src.capture import (
    CaptureError,
    capture_frame,
    describe_frame,
    is_valid_frame,
    run_preview,
    save_frame,
)


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)

    def read(self):
        return self.frames.pop(0) if self.frames else None


def bgr(width=640, height=480):
    return np.full((height, width, 3), 128, dtype=np.uint8)


@pytest.mark.parametrize("frame", [
    bgr(),
    np.zeros((10, 10), dtype=np.uint8),
])
def test_valid_frames(frame):
    assert is_valid_frame(frame)


@pytest.mark.parametrize("frame", [
    None,
    "image",
    np.zeros((0, 0, 3), dtype=np.uint8),
    np.zeros((10, 10, 4), dtype=np.uint8),
    np.zeros((10, 10, 3), dtype=np.float32),
    np.zeros(10, dtype=np.uint8),
])
def test_invalid_frames(frame):
    assert not is_valid_frame(frame)


def test_describe_bgr_frame():
    assert describe_frame(bgr(1280, 720)) == {
        "width": 1280, "height": 720, "channels": 3,
        "format": "BGR", "dtype": "uint8",
    }


def test_describe_gray_frame():
    info = describe_frame(np.zeros((480, 640), dtype=np.uint8))
    assert info["format"] == "GRAY" and info["channels"] == 1


def test_describe_invalid_frame_raises():
    with pytest.raises(CaptureError):
        describe_frame(None)


def test_capture_frame_skips_invalid_reads():
    frame = bgr()
    assert capture_frame(FakeCamera([None, frame])) is frame


def test_capture_frame_fails_after_attempts():
    with pytest.raises(CaptureError):
        capture_frame(FakeCamera([None, None, bgr()]), attempts=2)


def test_save_frame_roundtrip(tmp_path):
    frame = np.random.default_rng(0).integers(0, 256, (48, 64, 3), np.uint8)
    path = save_frame(frame, tmp_path / "sub", name="cube")
    assert path == tmp_path / "sub" / "cube.png"
    assert np.array_equal(cv2.imread(str(path)), frame)  # PNG sans perte


def test_save_frame_unique_default_names(tmp_path):
    first = save_frame(bgr(8, 8), tmp_path)
    second = save_frame(bgr(8, 8), tmp_path)
    assert first != second


def test_save_invalid_frame_raises(tmp_path):
    with pytest.raises(CaptureError):
        save_frame(None, tmp_path)


def test_save_write_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(CaptureError, match="Échec"):
        save_frame(bgr(), tmp_path)


def fake_gui(monkeypatch, keys, visible=1):
    """Simule la fenêtre OpenCV : `keys` est la suite des touches pressées."""
    keys = list(keys)
    calls = {"shown": 0, "destroyed": False}

    def imshow(name, frame):
        calls["shown"] += 1
        calls["last"] = frame

    def wait_key(delay):
        return ord(keys.pop(0)) if keys else -1

    def destroy(name):
        calls["destroyed"] = True

    gui = capture_module.cv2
    monkeypatch.setattr(gui, "namedWindow", lambda *a: None)
    monkeypatch.setattr(gui, "imshow", imshow)
    monkeypatch.setattr(gui, "waitKey", wait_key)
    monkeypatch.setattr(gui, "getWindowProperty", lambda *a: visible)
    monkeypatch.setattr(gui, "destroyWindow", destroy)
    return calls


def test_preview_captures_then_quits(monkeypatch, tmp_path):
    calls = fake_gui(monkeypatch, "xcq")
    saved = run_preview(FakeCamera([bgr()] * 3), tmp_path)
    assert len(saved) == 1 and saved[0].exists()
    assert calls["shown"] == 3 and calls["destroyed"]


def test_preview_overlay_is_displayed_but_raw_frame_saved(monkeypatch, tmp_path):
    calls = fake_gui(monkeypatch, "cq")
    raw = bgr(16, 16)
    marked = np.zeros_like(raw)
    saved = run_preview(FakeCamera([raw] * 2), tmp_path, overlay=lambda f: marked)
    assert calls["last"] is marked
    assert np.array_equal(cv2.imread(str(saved[0])), raw)


def test_preview_stops_when_window_closed(monkeypatch, tmp_path):
    calls = fake_gui(monkeypatch, "", visible=0)
    assert run_preview(FakeCamera([bgr()] * 5), tmp_path) == []
    assert calls["shown"] == 1 and calls["destroyed"]


def test_preview_stops_when_camera_fails(monkeypatch, tmp_path):
    calls = fake_gui(monkeypatch, "")
    assert run_preview(FakeCamera([]), tmp_path) == []
    assert calls["destroyed"]


@pytest.mark.hardware
def test_real_capture_and_save(tmp_path):
    if not list_cameras(max_index=1):
        pytest.skip("Aucune webcam disponible")
    with Camera() as cam:
        frame = capture_frame(cam)
    info = describe_frame(frame)
    assert info["format"] == "BGR" and info["width"] > 0
    assert save_frame(frame, tmp_path).exists()
