"""P1: target_dir から processing_dir への移動

スキャナーの書き込み途中・リネームを取りこぼさないことを、走査を手動で回して検証する。
"""

import os
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from service.file_move import TargetDirWatcher
from utils.config_manager import AppConfig


@pytest.fixture
def watcher(app_config: AppConfig, status_messages: list[str]) -> TargetDirWatcher:
    return TargetDirWatcher(app_config, status_messages.append)


def write_pdf(directory: str, name: str, content: bytes = b'%PDF-1.7 dummy') -> Path:
    path = Path(directory) / name
    path.write_bytes(content)
    return path


# --- 安定性チェック（P1） ---


def test_stable_file_moves_on_second_scan(
    watcher: TargetDirWatcher,
    app_config: AppConfig,
) -> None:
    """1回目の走査では記録のみ、変化がなければ2回目で移動する"""
    source = write_pdf(app_config.target_dir, 'scan.pdf')

    watcher.scan_once()
    assert source.exists()

    watcher.scan_once()
    assert not source.exists()
    assert (Path(app_config.processing_dir) / 'scan.pdf').read_bytes() == b'%PDF-1.7 dummy'


def test_growing_file_is_not_moved(
    watcher: TargetDirWatcher,
    app_config: AppConfig,
) -> None:
    """書き込み中でサイズが増え続けるファイルは移動しない"""
    source = write_pdf(app_config.target_dir, 'scan.pdf', b'%PDF')

    watcher.scan_once()
    source.write_bytes(b'%PDF-1.7 more data')
    watcher.scan_once()

    assert source.exists()

    watcher.scan_once()
    assert not source.exists()


def test_renamed_file_is_moved_after_stabilizing(
    watcher: TargetDirWatcher,
    app_config: AppConfig,
) -> None:
    """スキャナーがリネームしても、新しい名前で検出して移動する"""
    source = write_pdf(app_config.target_dir, 'temp.pdf')

    watcher.scan_once()
    renamed = Path(app_config.target_dir) / 'renamed.pdf'
    source.rename(renamed)

    watcher.scan_once()
    watcher.scan_once()

    assert not renamed.exists()
    assert (Path(app_config.processing_dir) / 'renamed.pdf').exists()


def test_non_pdf_files_stay_in_target_dir(
    watcher: TargetDirWatcher,
    app_config: AppConfig,
) -> None:
    note = Path(app_config.target_dir) / 'note.txt'
    note.write_text('text', encoding='utf-8')

    watcher.scan_once()
    watcher.scan_once()

    assert note.exists()
    assert os.listdir(app_config.processing_dir) == []


def test_uppercase_extension_is_moved(
    watcher: TargetDirWatcher,
    app_config: AppConfig,
) -> None:
    write_pdf(app_config.target_dir, 'SCAN.PDF')

    watcher.scan_once()
    watcher.scan_once()

    assert (Path(app_config.processing_dir) / 'SCAN.PDF').exists()


def test_existing_destination_is_overwritten(
    watcher: TargetDirWatcher,
    app_config: AppConfig,
) -> None:
    write_pdf(app_config.target_dir, 'scan.pdf', b'new')
    write_pdf(app_config.processing_dir, 'scan.pdf', b'old')

    watcher.scan_once()
    watcher.scan_once()

    assert (Path(app_config.processing_dir) / 'scan.pdf').read_bytes() == b'new'


# --- 通知とエラー処理（P2） ---


def test_move_notifies_status_callback(
    watcher: TargetDirWatcher,
    app_config: AppConfig,
    status_messages: list[str],
) -> None:
    write_pdf(app_config.target_dir, 'scan.pdf')

    watcher.scan_once()
    watcher.scan_once()

    assert len(status_messages) == 1
    assert 'scan.pdf' in status_messages[0]


def test_move_failure_is_retried_and_logged_once(
    watcher: TargetDirWatcher,
    app_config: AppConfig,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """移動できない間は再試行し、ログは最初の1回だけ出す"""
    write_pdf(app_config.target_dir, 'locked.pdf')
    move = mocker.patch('service.file_move.shutil.move', side_effect=OSError('locked'))

    with caplog.at_level('ERROR'):
        watcher.scan_once()
        watcher.scan_once()
        watcher.scan_once()

    assert move.call_count == 2
    assert len(caplog.records) == 1


def test_vanished_file_is_skipped(
    watcher: TargetDirWatcher,
    app_config: AppConfig,
    mocker: MockerFixture,
) -> None:
    """走査中に消えたファイルは無視する"""
    write_pdf(app_config.target_dir, 'scan.pdf')
    mocker.patch('service.file_move.os.stat', side_effect=OSError('gone'))

    watcher.scan_once()

    assert os.listdir(app_config.processing_dir) == []


def test_missing_target_dir_does_not_raise(
    watcher: TargetDirWatcher,
    app_config: AppConfig,
) -> None:
    os.rmdir(app_config.target_dir)

    watcher.scan_once()


# --- スレッドの起動と停止（P2） ---


def test_start_and_stop_thread(watcher: TargetDirWatcher) -> None:
    watcher.poll_interval = 0.01

    watcher.start()
    thread = watcher._thread
    assert thread is not None and thread.is_alive()

    watcher.stop()
    assert not thread.is_alive()


def test_start_is_idempotent(watcher: TargetDirWatcher) -> None:
    watcher.poll_interval = 0.01

    watcher.start()
    thread = watcher._thread
    watcher.start()

    assert watcher._thread is thread
    watcher.stop()


def test_stop_without_start_does_nothing(watcher: TargetDirWatcher) -> None:
    watcher.stop()

    assert watcher._thread is None
