"""Tests for cli._camera_arg positional camera parsing."""

import sys

import pytest

from ..cli import _camera_arg


@pytest.fixture
def restore_argv():
    old = sys.argv[:]
    yield
    sys.argv = old


def test_camera_arg_after_config_and_path(restore_argv):
    sys.argv = ["calibrate", "--config", "garden.toml", "1"]
    assert _camera_arg() == "1"


def test_camera_arg_after_short_config_and_path(restore_argv):
    sys.argv = ["calibrate", "-c", "garden.toml", "1"]
    assert _camera_arg() == "1"


def test_camera_arg_config_only(restore_argv):
    sys.argv = ["calibrate", "--config", "garden.toml"]
    assert _camera_arg() is None


def test_camera_arg_leading_index(restore_argv):
    sys.argv = ["calibrate", "1", "--config", "garden.toml"]
    assert _camera_arg() == "1"


def test_camera_arg_skips_explicit_camera_flag(restore_argv):
    sys.argv = ["measure", "--camera", "0", "--config", "x.toml"]
    assert _camera_arg() is None
