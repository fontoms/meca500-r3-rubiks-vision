import pytest

from src import camera as camera_module
from src.camera import Camera, CameraError, list_cameras


class FakeCapture:
    """Remplace cv2.VideoCapture pour tester sans matériel."""

    instances = []

    def __init__(self, index, backend=None, opened=True, grabs=True):
        self.index = index
        self._opened = opened
        # grabs : booléen, ou liste de résultats successifs de grab().
        self._grabs = list(grabs) if isinstance(grabs, list) else None
        self._always = grabs
        self.released = False
        self.settings = []
        FakeCapture.instances.append(self)

    def isOpened(self):
        return self._opened and not self.released

    def set(self, prop, value):
        self.settings.append((prop, value))
        return True

    def grab(self):
        if self._grabs is not None:
            return self._grabs.pop(0) if self._grabs else False
        return self._always

    def read(self):
        return (True, "frame") if self._always else (False, None)

    def get(self, prop):
        return 30.0

    def getBackendName(self):
        return "FAKE"

    def release(self):
        self.released = True


def install_fake(monkeypatch, opened=True, grabs=True, working=None):
    FakeCapture.instances = []

    def factory(index, backend=None):
        ok = opened if working is None else index in working
        return FakeCapture(index, backend, opened=ok, grabs=grabs)

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", factory)


def test_open_success(monkeypatch):
    install_fake(monkeypatch)
    cam = Camera().open()
    assert cam.is_opened
    assert cam.info()["backend"] == "FAKE"
    cam.release()
    assert not cam.is_opened
    assert FakeCapture.instances[0].released


def test_unavailable_camera_raises_and_releases(monkeypatch):
    install_fake(monkeypatch, opened=False)
    with pytest.raises(CameraError, match="introuvable"):
        Camera().open()
    assert FakeCapture.instances[0].released


def test_camera_without_frames_raises_and_releases(monkeypatch):
    install_fake(monkeypatch, grabs=False)
    with pytest.raises(CameraError, match="aucune image"):
        Camera().open()
    assert FakeCapture.instances[0].released


def test_warmup_retries_until_first_frame(monkeypatch):
    install_fake(monkeypatch, grabs=[False, False, True])
    with Camera(warmup_attempts=3) as cam:
        assert cam.is_opened


def test_warmup_gives_up_after_attempts(monkeypatch):
    install_fake(monkeypatch, grabs=[False, False, True])
    with pytest.raises(CameraError):
        Camera(warmup_attempts=2).open()


def test_opencv_error_is_wrapped(monkeypatch):
    def broken(index, backend=None):
        raise camera_module.cv2.error("boom")

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", broken)
    with pytest.raises(CameraError):
        Camera().open()


def test_context_manager_releases_even_on_error(monkeypatch):
    install_fake(monkeypatch)
    with pytest.raises(ZeroDivisionError):
        with Camera() as cam:
            assert cam.is_opened
            1 / 0
    assert not cam.is_opened
    assert FakeCapture.instances[0].released


def test_open_twice_reuses_capture(monkeypatch):
    install_fake(monkeypatch)
    with Camera() as cam:
        cam.open()
    assert len(FakeCapture.instances) == 1


def test_release_without_open_is_safe():
    Camera().release()


def test_read_returns_frame(monkeypatch):
    install_fake(monkeypatch)
    with Camera() as cam:
        assert cam.read() == "frame"


def test_read_returns_none_on_failure(monkeypatch):
    install_fake(monkeypatch)
    with Camera() as cam:
        cam._capture._always = False
        assert cam.read() is None


def test_read_requires_open_camera():
    with pytest.raises(CameraError):
        Camera().read()


def test_info_requires_open_camera():
    with pytest.raises(CameraError):
        Camera().info()


@pytest.mark.parametrize("index", [-1, "0", 1.5])
def test_invalid_index(index):
    with pytest.raises(ValueError):
        Camera(index)


def test_list_cameras(monkeypatch):
    install_fake(monkeypatch, working={0, 2})
    assert list_cameras(max_index=4) == [0, 2]
    assert all(c.released for c in FakeCapture.instances)


@pytest.mark.hardware
def test_real_webcam():
    if not list_cameras(max_index=2):
        pytest.skip("Aucune webcam disponible")
    with Camera() as cam:
        info = cam.info()
        assert info["width"] > 0 and info["height"] > 0
    assert not cam.is_opened


def test_automatic_controls_untouched_by_default(monkeypatch):
    install_fake(monkeypatch)
    with Camera():
        props = [p for p, _ in FakeCapture.instances[0].settings]
    assert camera_module.cv2.CAP_PROP_EXPOSURE not in props
    assert camera_module.cv2.CAP_PROP_AUTO_WB not in props


def test_manual_exposure_and_locked_white_balance(monkeypatch):
    install_fake(monkeypatch)
    with Camera(exposure=-5, lock_white_balance=True):
        settings = FakeCapture.instances[0].settings
    cv2 = camera_module.cv2
    assert (cv2.CAP_PROP_AUTO_EXPOSURE, camera_module.MANUAL_EXPOSURE_MODE) in settings
    assert (cv2.CAP_PROP_EXPOSURE, -5) in settings
    assert (cv2.CAP_PROP_AUTO_WB, 0) in settings


def test_manual_controls_use_directshow_on_windows(monkeypatch):
    monkeypatch.setattr(camera_module.sys, "platform", "win32")
    cv2 = camera_module.cv2
    assert Camera(exposure=-5).backend == cv2.CAP_DSHOW
    assert Camera(lock_white_balance=True).backend == cv2.CAP_DSHOW
    assert Camera().backend == camera_module.DEFAULT_BACKEND
