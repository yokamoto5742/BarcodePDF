"""P1: 保存期間を過ぎたエラーPDFの削除"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from service.error_pdf_cleanup import cleanup_error_pdfs
from utils.config_manager import AppConfig


def touch_file(directory: str, name: str, days_old: float = 0) -> Path:
    path = Path(directory) / name
    path.write_bytes(b'%PDF-1.7 dummy')
    timestamp = (datetime.now() - timedelta(days=days_old)).timestamp()
    os.utime(path, (timestamp, timestamp))
    return path


def test_removes_expired_pdf(app_config: AppConfig) -> None:
    expired = touch_file(app_config.error_dir, 'old.pdf', days_old=30)

    cleanup_error_pdfs(app_config)

    assert not expired.exists()


def test_keeps_recent_pdf(app_config: AppConfig) -> None:
    recent = touch_file(app_config.error_dir, 'new.pdf', days_old=1)

    cleanup_error_pdfs(app_config)

    assert recent.exists()


def test_ignores_non_pdf_files(app_config: AppConfig) -> None:
    other = touch_file(app_config.error_dir, 'memo.txt', days_old=30)

    cleanup_error_pdfs(app_config)

    assert other.exists()


def test_matches_extension_case_insensitively(app_config: AppConfig) -> None:
    expired = touch_file(app_config.error_dir, 'OLD.PDF', days_old=30)

    cleanup_error_pdfs(app_config)

    assert not expired.exists()


def test_skips_subdirectories(app_config: AppConfig) -> None:
    """拡張子が一致するフォルダを削除対象にしない"""
    sub_directory = Path(app_config.error_dir) / 'archive.pdf'
    sub_directory.mkdir()

    cleanup_error_pdfs(app_config)

    assert sub_directory.is_dir()


def test_does_nothing_for_missing_directory(
    app_config: AppConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app_config.error_dir = 'C:/no/such/directory'

    cleanup_error_pdfs(app_config)

    assert caplog.records == []


def test_continues_after_remove_error(
    app_config: AppConfig,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    touch_file(app_config.error_dir, 'one.pdf', days_old=30)
    touch_file(app_config.error_dir, 'two.pdf', days_old=30)
    mocker.patch('utils.file_cleanup.os.remove', side_effect=OSError('使用中'))

    cleanup_error_pdfs(app_config)

    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 2


def test_uses_configured_retention_days(app_config: AppConfig) -> None:
    """LOGGING/log_retention_days をエラーPDFの保存期間として使う"""
    app_config.log_retention_days = 30
    kept = touch_file(app_config.error_dir, 'old.pdf', days_old=10)

    cleanup_error_pdfs(app_config)

    assert kept.exists()
