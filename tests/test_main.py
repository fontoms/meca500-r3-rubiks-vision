import pytest

from main import parse_args


def test_default_camera_is_auto():
    assert parse_args([]).camera is None


def test_camera_option():
    assert parse_args(["--camera", "2"]).camera == 2


def test_camera_option_rejects_non_integer():
    with pytest.raises(SystemExit):
        parse_args(["--camera", "abc"])


def test_image_option():
    assert parse_args(["--image", "cube.png"]).image == "cube.png"


def test_unreadable_image_returns_error(tmp_path):
    from main import run_on_image
    assert run_on_image(str(tmp_path / "absent.png")) == 1
