import os
import pytest


def test_config_is_dict(config):
    assert isinstance(config, dict)


def test_required_keys_exist(config):
    required_keys = [
        "PROJECT_ROOT",
        "SOURCE_DIR",
        "PACKAGE_DIR",
        "DATASET_ROOT",
        "BACKBONE_DIR",
        "OUTPUT_ROOT",
        "CHECKPOINT_DIR",
        "RESULT_DIR",
        "LOG_DIR",
    ]

    for key in required_keys:
        assert key in config, f"Missing required config key: {key}"



def test_dataset_root_absolute_and_fixed(config):
    assert config["DATASET_ROOT"] == "/home/namu/myspace/NAMU/datasets"
    assert os.path.isabs(config["DATASET_ROOT"])


def test_backbone_dir_absolute_and_fixed(config):
    assert config["BACKBONE_DIR"] == "/home/namu/myspace/NAMU/backbones"
    assert os.path.isabs(config["BACKBONE_DIR"])


def test_project_root_matches_pwd(config):
    assert config["PROJECT_ROOT"] == os.getcwd()


def test_all_paths_are_absolute(config):
    for key, value in config.items():
        assert isinstance(value, str)
        assert os.path.isabs(value), f"Path is not absolute: {key}={value}"


def test_output_subdirs_relationship(config):
    assert config["CHECKPOINT_DIR"].startswith(config["OUTPUT_ROOT"])
    assert config["RESULT_DIR"].startswith(config["OUTPUT_ROOT"])
    assert config["LOG_DIR"].startswith(config["OUTPUT_ROOT"])


def test_source_package_relationship(config):
    assert config["PACKAGE_DIR"].startswith(config["SOURCE_DIR"])
