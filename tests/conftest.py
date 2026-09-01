"""全テスト共通のfixture"""

import logging
from pathlib import Path

import pytest

from utils.config_manager import AppConfig

CONFIG_TEMPLATE = """[Directories]
processing_dir = {processing_dir}
error_dir = {error_dir}
done_dir = {done_dir}

[UI]
width = 600
height = 500

[Options]
auto_open_error_folder = False
start_minimized = True

[LOGGING]
log_retention_days = 7
log_directory = {log_dir}
log_level = INFO
debug_mode = False
project_name = BarcodePDF
"""


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """作業フォルダを実作成し、それらを指すconfig.iniのパスを返す"""
    for name in ('processing', 'error', 'done', 'log'):
        (tmp_path / name).mkdir()

    path = tmp_path / 'config.ini'
    path.write_text(
        CONFIG_TEMPLATE.format(
            processing_dir=tmp_path / 'processing',
            error_dir=tmp_path / 'error',
            done_dir=tmp_path / 'done',
            log_dir=tmp_path / 'log',
        ),
        encoding='utf-8',
    )
    return path


@pytest.fixture
def app_config(config_file: Path) -> AppConfig:
    return AppConfig(config_file)


@pytest.fixture
def status_messages() -> list[str]:
    """status_callback が受け取ったメッセージを蓄積するリスト（`.append` を渡して使う）"""
    return []


@pytest.fixture
def pdf_in_processing(app_config: AppConfig) -> Path:
    """処理フォルダに置かれたPDFファイル（中身の妥当性は問わない）"""
    path = Path(app_config.processing_dir) / 'input.pdf'
    path.write_bytes(b'%PDF-1.7 dummy')
    return path


@pytest.fixture(autouse=True)
def restore_root_logger() -> object:
    """setup_logging がルートロガーのハンドラを全除去するため、テストごとに復元する"""
    root_logger = logging.getLogger()
    saved_handlers = root_logger.handlers[:]
    saved_level = root_logger.level

    yield

    for handler in root_logger.handlers[:]:
        if handler not in saved_handlers:
            root_logger.removeHandler(handler)
            # tmp_path を削除できるようログファイルのハンドルを解放する
            if isinstance(handler, logging.FileHandler):
                handler.close()

    root_logger.handlers[:] = saved_handlers
    root_logger.setLevel(saved_level)
