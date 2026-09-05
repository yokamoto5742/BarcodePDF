"""P1/P2: ロギング初期化と古いログの削除"""

import configparser
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from utils.log_rotation import (
    cleanup_error_pdfs,
    cleanup_old_logs,
    get_log_info,
    setup_debug_logging,
    setup_logging,
)

PROJECT_ROOT = Path(__file__).parents[2]


def make_config(error_dir: str | None = None, **overrides: str) -> configparser.ConfigParser:
    values = {
        'log_directory': 'logs',
        'log_retention_days': '7',
        'project_name': 'BarcodePDF',
        'log_level': 'INFO',
        'debug_mode': 'False',
    }
    values.update(overrides)

    config = configparser.ConfigParser()
    config['LOGGING'] = values
    if error_dir is not None:
        config['Directories'] = {'error_dir': error_dir}
    return config


def touch_log(directory: Path, name: str, days_old: float = 0) -> Path:
    path = directory / name
    path.write_text('log', encoding='utf-8')
    timestamp = (datetime.now() - timedelta(days=days_old)).timestamp()
    os.utime(path, (timestamp, timestamp))
    return path


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    directory = tmp_path / 'log'
    directory.mkdir()
    return directory


@pytest.fixture
def error_dir(tmp_path: Path) -> Path:
    directory = tmp_path / 'error'
    directory.mkdir()
    return directory


@pytest.fixture
def debug_logger_cleanup() -> object:
    """setup_debug_logging が 'debug' ロガーに残すハンドラを片付ける"""
    yield
    debug_logger = logging.getLogger('debug')
    for handler in debug_logger.handlers[:]:
        debug_logger.removeHandler(handler)
        handler.close()


# --- setup_logging（P1） ---


def test_setup_logging_creates_log_file(log_dir: Path) -> None:
    setup_logging(make_config(log_directory=str(log_dir)))

    assert (log_dir / 'BarcodePDF.log').exists()


def test_setup_logging_creates_missing_directory(tmp_path: Path) -> None:
    target = tmp_path / 'not_yet'

    setup_logging(make_config(log_directory=str(target)))

    assert target.is_dir()


def test_setup_logging_registers_file_and_console_handlers(log_dir: Path) -> None:
    setup_logging(make_config(log_directory=str(log_dir)))

    handlers = logging.getLogger().handlers
    assert len(handlers) == 2
    assert isinstance(handlers[0], logging.FileHandler)
    assert handlers[0].suffix == '%Y-%m-%d.log'  # type: ignore[attr-defined]
    assert handlers[1].level == logging.WARNING


def test_setup_logging_does_not_duplicate_handlers(log_dir: Path) -> None:
    config = make_config(log_directory=str(log_dir))

    setup_logging(config)
    setup_logging(config)

    assert len(logging.getLogger().handlers) == 2


def test_setup_logging_resolves_relative_directory(mocker: MockerFixture) -> None:
    handler = mocker.patch('utils.log_rotation.TimedRotatingFileHandler')
    handler.return_value.level = logging.NOTSET
    mocker.patch('utils.log_rotation.os.makedirs')
    mocker.patch('utils.log_rotation.os.path.exists', return_value=True)
    mocker.patch('utils.log_rotation.cleanup_old_logs')

    setup_logging(make_config(log_directory='logs'))

    expected = str(PROJECT_ROOT / 'logs' / 'BarcodePDF.log')
    assert handler.call_args.kwargs['filename'] == expected


def test_setup_logging_applies_configured_level(log_dir: Path) -> None:
    setup_logging(make_config(log_directory=str(log_dir), log_level='DEBUG'))

    assert logging.getLogger().level == logging.DEBUG


@pytest.mark.parametrize('level', ['INVALID', 'Formatter'])
def test_setup_logging_falls_back_to_info_for_bad_level(level: str, log_dir: Path) -> None:
    """'Formatter' のような logging に実在する非レベル属性でも INFO に落とす"""
    setup_logging(make_config(log_directory=str(log_dir), log_level=level))

    assert logging.getLogger().level == logging.INFO


def test_setup_logging_uses_retention_days_as_backup_count(log_dir: Path) -> None:
    setup_logging(make_config(log_directory=str(log_dir), log_retention_days='3'))

    handler = logging.getLogger().handlers[0]
    assert handler.backupCount == 3  # type: ignore[attr-defined]


def test_setup_logging_raises_on_permission_error(mocker: MockerFixture, tmp_path: Path) -> None:
    mocker.patch('utils.log_rotation.os.makedirs', side_effect=PermissionError('拒否'))

    with pytest.raises(PermissionError, match='ログディレクトリの作成権限がありません'):
        setup_logging(make_config(log_directory=str(tmp_path / 'denied')))


def test_setup_logging_wraps_unexpected_error(mocker: MockerFixture, log_dir: Path) -> None:
    mocker.patch('utils.log_rotation.TimedRotatingFileHandler', side_effect=ValueError('不正'))

    with pytest.raises(Exception, match='ログ設定の初期化中にエラーが発生しました'):
        setup_logging(make_config(log_directory=str(log_dir)))


def test_setup_logging_loads_config_when_omitted(mocker: MockerFixture, log_dir: Path) -> None:
    load = mocker.patch(
        'utils.log_rotation.load_config', return_value=make_config(log_directory=str(log_dir))
    )

    setup_logging()

    load.assert_called_once()


# --- cleanup_old_logs（P1: ファイル削除を伴う） ---


def test_cleanup_removes_expired_rotated_logs(log_dir: Path) -> None:
    expired = touch_log(log_dir, 'BarcodePDF.log.2020-01-01.log', days_old=30)

    cleanup_old_logs(str(log_dir), 7, 'BarcodePDF')

    assert not expired.exists()


def test_cleanup_keeps_recent_rotated_logs(log_dir: Path) -> None:
    recent = touch_log(log_dir, 'BarcodePDF.log.2020-01-01.log', days_old=1)

    cleanup_old_logs(str(log_dir), 7, 'BarcodePDF')

    assert recent.exists()


def test_cleanup_never_removes_current_log(log_dir: Path) -> None:
    current = touch_log(log_dir, 'BarcodePDF.log', days_old=365)

    cleanup_old_logs(str(log_dir), 7, 'BarcodePDF')

    assert current.exists()


@pytest.mark.parametrize('filename', [
    'other.log.2020-01-01.log',
    'BarcodePDF.log.2020-01-01.txt',
    'BarcodePDF.log.20200101.log',
    'debug.log',
])
def test_cleanup_ignores_unrelated_files(filename: str, log_dir: Path) -> None:
    unrelated = touch_log(log_dir, filename, days_old=365)

    cleanup_old_logs(str(log_dir), 7, 'BarcodePDF')

    assert unrelated.exists()


def test_cleanup_removes_log_aged_exactly_retention_days(log_dir: Path) -> None:
    """保持期間ちょうど（>=判定）は削除される"""
    boundary = touch_log(log_dir, 'BarcodePDF.log.2020-01-01.log', days_old=7)

    cleanup_old_logs(str(log_dir), 7, 'BarcodePDF')

    assert not boundary.exists()


def test_cleanup_removes_all_rotated_logs_when_retention_is_zero(log_dir: Path) -> None:
    fresh = touch_log(log_dir, 'BarcodePDF.log.2020-01-01.log', days_old=0)

    cleanup_old_logs(str(log_dir), 0, 'BarcodePDF')

    assert not fresh.exists()


def test_cleanup_continues_after_remove_error(
    log_dir: Path,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    touch_log(log_dir, 'BarcodePDF.log.2020-01-01.log', days_old=30)
    touch_log(log_dir, 'BarcodePDF.log.2020-01-02.log', days_old=30)
    mocker.patch('utils.log_rotation.os.remove', side_effect=OSError('使用中'))

    cleanup_old_logs(str(log_dir), 7, 'BarcodePDF')

    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 2


def test_cleanup_logs_error_for_missing_directory(caplog: pytest.LogCaptureFixture) -> None:
    cleanup_old_logs('C:/no/such/directory', 7, 'BarcodePDF')

    assert caplog.records[0].levelno == logging.ERROR


# --- cleanup_error_pdfs（P1: ファイル削除を伴う） ---


def test_cleanup_error_pdfs_removes_expired_pdf(error_dir: Path) -> None:
    expired = touch_log(error_dir, 'old.pdf', days_old=30)

    cleanup_error_pdfs(make_config(error_dir=str(error_dir)), 7)

    assert not expired.exists()


def test_cleanup_error_pdfs_keeps_recent_pdf(error_dir: Path) -> None:
    recent = touch_log(error_dir, 'new.pdf', days_old=1)

    cleanup_error_pdfs(make_config(error_dir=str(error_dir)), 7)

    assert recent.exists()


def test_cleanup_error_pdfs_ignores_non_pdf_files(error_dir: Path) -> None:
    other = touch_log(error_dir, 'memo.txt', days_old=30)

    cleanup_error_pdfs(make_config(error_dir=str(error_dir)), 7)

    assert other.exists()


def test_cleanup_error_pdfs_matches_extension_case_insensitively(error_dir: Path) -> None:
    expired = touch_log(error_dir, 'OLD.PDF', days_old=30)

    cleanup_error_pdfs(make_config(error_dir=str(error_dir)), 7)

    assert not expired.exists()


def test_cleanup_error_pdfs_skips_subdirectories(error_dir: Path) -> None:
    """拡張子が一致するフォルダを削除対象にしない"""
    sub_directory = error_dir / 'archive.pdf'
    sub_directory.mkdir()

    cleanup_error_pdfs(make_config(error_dir=str(error_dir)), 7)

    assert sub_directory.is_dir()


def test_cleanup_error_pdfs_does_nothing_without_directories_section(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cleanup_error_pdfs(make_config(), 7)

    assert caplog.records == []


def test_cleanup_error_pdfs_does_nothing_for_missing_directory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cleanup_error_pdfs(make_config(error_dir='C:/no/such/directory'), 7)

    assert caplog.records == []


def test_cleanup_error_pdfs_continues_after_remove_error(
    error_dir: Path,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    touch_log(error_dir, 'one.pdf', days_old=30)
    touch_log(error_dir, 'two.pdf', days_old=30)
    mocker.patch('utils.log_rotation.os.remove', side_effect=OSError('使用中'))

    cleanup_error_pdfs(make_config(error_dir=str(error_dir)), 7)

    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 2


def test_setup_logging_cleans_up_error_pdfs(log_dir: Path, error_dir: Path) -> None:
    """ログローテーションと同じタイミングでエラーPDFも削除する"""
    expired = touch_log(error_dir, 'old.pdf', days_old=30)

    setup_logging(make_config(log_directory=str(log_dir), error_dir=str(error_dir)))

    assert not expired.exists()


# --- setup_debug_logging（P2） ---


def test_setup_debug_logging_returns_none_when_disabled() -> None:
    assert setup_debug_logging(make_config(debug_mode='False')) is None


def test_setup_debug_logging_creates_logger(
    log_dir: Path,
    debug_logger_cleanup: None,
) -> None:
    debug_logger = setup_debug_logging(
        make_config(debug_mode='True', log_directory=str(log_dir))
    )

    assert debug_logger is not None
    assert debug_logger.level == logging.DEBUG
    assert debug_logger.propagate is False
    assert (log_dir / 'debug.log').exists()


def test_setup_debug_logging_resolves_relative_directory(mocker: MockerFixture) -> None:
    file_handler = mocker.patch('utils.log_rotation.logging.FileHandler')
    file_handler.return_value.level = logging.NOTSET

    setup_debug_logging(make_config(debug_mode='True', log_directory='logs'))

    assert file_handler.call_args.args[0] == str(PROJECT_ROOT / 'logs' / 'debug.log')


def test_setup_debug_logging_loads_config_when_omitted(mocker: MockerFixture) -> None:
    load = mocker.patch('utils.log_rotation.load_config', return_value=make_config())

    setup_debug_logging()

    load.assert_called_once()


def test_setup_debug_logging_returns_none_on_error(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mocker.patch('utils.log_rotation.logging.FileHandler', side_effect=OSError('作成失敗'))

    assert setup_debug_logging(make_config(debug_mode='True', log_directory='C:/x')) is None
    assert caplog.records[-1].levelno == logging.ERROR


# --- get_log_info（P2） ---


def test_get_log_info_returns_absolute_paths(log_dir: Path) -> None:
    info = get_log_info(make_config(log_directory=str(log_dir), project_name='BarcodePDF'))

    assert info is not None
    assert info['log_directory'] == str(log_dir)
    assert info['main_log_file'] == str(log_dir / 'BarcodePDF.log')
    assert info['log_retention_days'] == 7
    assert info['debug_mode'] is False
    assert info['debug_log_file'] is None


def test_get_log_info_includes_debug_log_when_enabled(log_dir: Path) -> None:
    info = get_log_info(make_config(log_directory=str(log_dir), debug_mode='True'))

    assert info is not None
    assert info['debug_log_file'] == str(log_dir / 'debug.log')


def test_get_log_info_resolves_relative_directory() -> None:
    info = get_log_info(make_config(log_directory='logs'))

    assert info is not None
    assert info['log_directory'] == str(PROJECT_ROOT / 'logs')


def test_get_log_info_loads_config_when_omitted(mocker: MockerFixture) -> None:
    load = mocker.patch('utils.log_rotation.load_config', return_value=make_config())

    get_log_info()

    load.assert_called_once()


def test_get_log_info_returns_none_on_error(mocker: MockerFixture) -> None:
    mocker.patch('utils.log_rotation.get_config_value', side_effect=RuntimeError('失敗'))

    assert get_log_info(make_config()) is None
