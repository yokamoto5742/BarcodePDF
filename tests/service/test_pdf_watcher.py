"""P1: target_dir の監視とバーコード処理への受け渡し

スキャナーの書き込み途中・リネームを取りこぼさないことを、走査を手動で回して検証する。
"""

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from service.pdf_watcher import PdfWatcher
from utils.config_manager import AppConfig


@pytest.fixture
def process_pdf(mocker: MockerFixture) -> MagicMock:
    """バーコード読み取りは別テストで検証するため、呼び出しの有無だけを見る"""
    return mocker.patch('service.pdf_watcher.process_pdf')


@pytest.fixture
def watcher(app_config: AppConfig, status_messages: list[str]) -> PdfWatcher:
    return PdfWatcher(app_config, status_messages.append)


def write_pdf(directory: str, name: str, content: bytes = b'%PDF-1.7 dummy') -> Path:
    path = Path(directory) / name
    path.write_bytes(content)
    return path


def processed_paths(process_pdf: MagicMock) -> list[str]:
    return [call.args[0] for call in process_pdf.call_args_list]


# --- 安定性チェック（P1） ---


def test_stable_file_is_processed_on_second_scan(
    watcher: PdfWatcher,
    app_config: AppConfig,
    process_pdf: MagicMock,
) -> None:
    """1回目の走査では記録のみ、変化がなければ2回目で処理する"""
    source = write_pdf(app_config.target_dir, 'scan.pdf')

    watcher.scan_once()
    assert processed_paths(process_pdf) == []

    watcher.scan_once()
    assert processed_paths(process_pdf) == [str(source)]


def test_growing_file_is_not_processed(
    watcher: PdfWatcher,
    app_config: AppConfig,
    process_pdf: MagicMock,
) -> None:
    """書き込み中でサイズが増え続けるファイルは処理しない"""
    source = write_pdf(app_config.target_dir, 'scan.pdf', b'%PDF')

    watcher.scan_once()
    source.write_bytes(b'%PDF-1.7 more data')
    watcher.scan_once()

    assert processed_paths(process_pdf) == []

    watcher.scan_once()
    assert processed_paths(process_pdf) == [str(source)]


def test_renamed_file_is_processed_after_stabilizing(
    watcher: PdfWatcher,
    app_config: AppConfig,
    process_pdf: MagicMock,
) -> None:
    """スキャナーがリネームしても、新しい名前で検出して処理する"""
    source = write_pdf(app_config.target_dir, 'temp.pdf')

    watcher.scan_once()
    renamed = Path(app_config.target_dir) / 'renamed.pdf'
    source.rename(renamed)

    watcher.scan_once()
    watcher.scan_once()

    assert processed_paths(process_pdf) == [str(renamed)]


def test_non_pdf_files_are_ignored(
    watcher: PdfWatcher,
    app_config: AppConfig,
    process_pdf: MagicMock,
) -> None:
    note = Path(app_config.target_dir) / 'note.txt'
    note.write_text('text', encoding='utf-8')

    watcher.scan_once()
    watcher.scan_once()

    assert note.exists()
    assert processed_paths(process_pdf) == []


def test_uppercase_extension_is_processed(
    watcher: PdfWatcher,
    app_config: AppConfig,
    process_pdf: MagicMock,
) -> None:
    source = write_pdf(app_config.target_dir, 'SCAN.PDF')

    watcher.scan_once()
    watcher.scan_once()

    assert processed_paths(process_pdf) == [str(source)]


def test_file_left_behind_is_retried_after_stabilizing(
    watcher: PdfWatcher,
    app_config: AppConfig,
    process_pdf: MagicMock,
) -> None:
    """処理できずに残ったファイルは、次に安定した時点で再試行する"""
    source = write_pdf(app_config.target_dir, 'locked.pdf')

    watcher.scan_once()
    watcher.scan_once()
    watcher.scan_once()
    watcher.scan_once()

    assert processed_paths(process_pdf) == [str(source), str(source)]


# --- エラー処理（P2） ---


def test_vanished_file_is_skipped(
    watcher: PdfWatcher,
    app_config: AppConfig,
    mocker: MockerFixture,
    process_pdf: MagicMock,
) -> None:
    """走査中に消えたファイルは無視する"""
    write_pdf(app_config.target_dir, 'scan.pdf')
    mocker.patch('service.pdf_watcher.os.stat', side_effect=OSError('gone'))

    watcher.scan_once()

    assert processed_paths(process_pdf) == []


def test_missing_target_dir_is_logged_without_raising(
    watcher: PdfWatcher,
    app_config: AppConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    os.rmdir(app_config.target_dir)

    with caplog.at_level('ERROR'):
        watcher.scan_once()

    assert len(caplog.records) == 1


# --- スレッドの起動と停止（P2） ---


def test_start_and_stop_thread(watcher: PdfWatcher, mocker: MockerFixture) -> None:
    """スレッドが走査を繰り返し、stop で確実に終了する"""
    watcher.poll_interval = 0.01
    scanned = threading.Event()
    mocker.patch.object(watcher, 'scan_once', side_effect=lambda: scanned.set())

    watcher.start()
    thread = watcher._thread
    assert thread is not None and thread.is_alive()
    assert scanned.wait(timeout=5)

    watcher.stop()
    assert not thread.is_alive()


def test_start_is_idempotent(watcher: PdfWatcher) -> None:
    watcher.poll_interval = 0.01

    watcher.start()
    thread = watcher._thread
    watcher.start()

    assert watcher._thread is thread
    watcher.stop()


def test_stop_without_start_does_nothing(watcher: PdfWatcher) -> None:
    watcher.stop()

    assert watcher._thread is None
