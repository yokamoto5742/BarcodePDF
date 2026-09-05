"""P0/P2: config.ini の読み書き"""

import configparser
import sys
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from utils.config_manager import (
    AppConfig,
    ConfigManager,
    get_config_path,
    load_config,
)

MINIMAL_INI = """[Directories]
target_dir = C:\\target
error_dir = C:\\err
done_dir = C:\\done
"""


BARCODE_INI = """
[Barcode]
contrast_factor = 3.0
render_zoom = 2.5
top_band_ratio = 0.3
min_barcode_width_ratio = 0.4
"""


def write_ini(path: Path, text: str) -> Path:
    path.write_text(text, encoding='utf-8')
    return path


# --- ConfigManager.load_config（P0） ---


def test_load_config_reads_utf8_file(config_file: Path) -> None:
    manager = ConfigManager(config_file)

    assert manager.config.get('LOGGING', 'project_name') == 'BarcodePDF'


def test_load_config_raises_when_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / 'missing.ini'

    with pytest.raises(FileNotFoundError, match='missing.ini'):
        ConfigManager(missing)


def test_load_config_falls_back_to_cp932(tmp_path: Path) -> None:
    path = tmp_path / 'cp932.ini'
    path.write_bytes('[LOGGING]\nproject_name = 日本語設定\n'.encode('cp932'))

    manager = ConfigManager(path)

    assert manager.config.get('LOGGING', 'project_name') == '日本語設定'


def test_load_config_raises_when_encoding_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / 'broken.ini'
    # UTF-8でもCP932でもデコードできないバイト列
    path.write_bytes(b'[Directories]\nprocessing_dir = \x81\x20\x82\x00\n')

    with pytest.raises(OSError, match='Failed to load config'):
        ConfigManager(path)


def test_ensure_section_creates_missing_section(tmp_path: Path) -> None:
    manager = ConfigManager(write_ini(tmp_path / 'c.ini', MINIMAL_INI))

    manager.ensure_section('Options')

    assert 'Options' in manager.config


def test_ensure_section_keeps_existing_values(config_file: Path) -> None:
    manager = ConfigManager(config_file)

    manager.ensure_section('Options')

    assert manager.config.getboolean('Options', 'auto_open_error_folder') is False


# --- ConfigManager.save_config（P2） ---


def test_save_config_writes_file(config_file: Path) -> None:
    manager = ConfigManager(config_file)
    manager.config['Options']['auto_open_error_folder'] = 'True'

    manager.save_config()

    assert ConfigManager(config_file).config.getboolean('Options', 'auto_open_error_folder') is True


def test_save_config_reports_save_failure(config_file: Path, mocker: MockerFixture) -> None:
    mocker.patch('builtins.open', side_effect=OSError('書き込み不可'))
    manager = ConfigManager(config_file)

    with pytest.raises(OSError, match='Failed to save config'):
        manager.save_config()


# --- AppConfig（P0） ---


def test_app_config_reads_typed_values(app_config: AppConfig, tmp_path: Path) -> None:
    assert app_config.target_dir == str(tmp_path / 'target')
    assert app_config.error_dir == str(tmp_path / 'error')
    assert app_config.done_dir == str(tmp_path / 'done')
    assert app_config.ui_width == 600
    assert app_config.ui_height == 500
    assert app_config.auto_open_error_folder is False
    assert app_config.log_retention_days == 7


def test_app_config_reads_barcode_values(tmp_path: Path) -> None:
    config = AppConfig(write_ini(tmp_path / 'c.ini', MINIMAL_INI + BARCODE_INI))

    assert config.contrast_factor == 3.0
    assert config.render_zoom == 2.5
    assert config.top_band_ratio == 0.3
    assert config.min_barcode_width_ratio == 0.4


def test_app_config_uses_fallbacks_for_optional_values(tmp_path: Path) -> None:
    config = AppConfig(write_ini(tmp_path / 'c.ini', MINIMAL_INI))

    assert config.log_dir == 'logs'
    assert config.log_retention_days == 7
    assert config.ui_width == 600
    assert config.ui_height == 500
    assert config.auto_open_error_folder is True
    assert config.contrast_factor == 2.0
    assert config.render_zoom == 2.0
    assert config.top_band_ratio == 0.15
    assert config.min_barcode_width_ratio == 0.20


def test_app_config_raises_when_directories_section_missing(tmp_path: Path) -> None:
    with pytest.raises(configparser.NoSectionError):
        AppConfig(write_ini(tmp_path / 'c.ini', '[UI]\nwidth = 100\n'))


def test_app_config_raises_when_directory_key_missing(tmp_path: Path) -> None:
    with pytest.raises(configparser.NoOptionError):
        AppConfig(write_ini(tmp_path / 'c.ini', '[Directories]\nerror_dir = C:\\err\n'))


# --- AppConfig.ensure_directories（P0） ---


def test_ensure_directories_creates_missing_directories(
    app_config: AppConfig,
    tmp_path: Path,
) -> None:
    (tmp_path / 'done').rmdir()
    (tmp_path / 'error').rmdir()

    created = app_config.ensure_directories()

    assert sorted(created) == sorted([app_config.error_dir, app_config.done_dir])
    assert Path(app_config.done_dir).is_dir()


def test_ensure_directories_returns_empty_when_all_exist(app_config: AppConfig) -> None:
    assert app_config.ensure_directories() == []


def test_ensure_directories_keeps_existing_files(app_config: AppConfig) -> None:
    existing = Path(app_config.done_dir) / 'keep.pdf'
    existing.write_bytes(b'keep')

    app_config.ensure_directories()

    assert existing.read_bytes() == b'keep'


def test_ensure_directories_creates_nested_path(app_config: AppConfig, tmp_path: Path) -> None:
    app_config.done_dir = str(tmp_path / 'a' / 'b' / 'c')

    created = app_config.ensure_directories()

    assert created == [app_config.done_dir]
    assert Path(app_config.done_dir).is_dir()


# --- AppConfig.save（P0） ---


def test_save_round_trips_values(app_config: AppConfig, config_file: Path, tmp_path: Path) -> None:
    app_config.target_dir = str(tmp_path / 'new_target')
    app_config.log_dir = str(tmp_path / 'new_log')
    app_config.auto_open_error_folder = True

    app_config.save()

    reloaded = AppConfig(config_file)
    assert reloaded.target_dir == str(tmp_path / 'new_target')
    assert reloaded.log_dir == str(tmp_path / 'new_log')
    assert reloaded.auto_open_error_folder is True


def test_save_creates_missing_sections(tmp_path: Path) -> None:
    path = write_ini(tmp_path / 'c.ini', MINIMAL_INI)
    config = AppConfig(path)

    config.save()

    reloaded = AppConfig(path)
    assert reloaded.log_dir == 'logs'
    assert reloaded.auto_open_error_folder is True


# --- get_config_path / load_config（P2） ---


def test_get_config_path_uses_module_directory() -> None:
    assert get_config_path() == Path(__file__).parents[2] / 'utils' / 'config.ini'


def test_get_config_path_uses_meipass_when_frozen(mocker: MockerFixture) -> None:
    mocker.patch.object(sys, 'frozen', True, create=True)
    mocker.patch.object(sys, '_MEIPASS', r'C:\dist', create=True)

    assert get_config_path() == Path(r'C:\dist') / 'config.ini'


def test_load_config_returns_parser_with_directories() -> None:
    assert load_config().has_section('Directories')

