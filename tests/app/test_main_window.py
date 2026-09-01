"""P1/P2: GUIの振る舞い

tkinterを実起動せず、__init__を通さずに生成したインスタンスへ依存を注入して検証する。
"""

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import cast
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from app import __version__
from app.main_window import PDFProcessorApp
from utils.config_manager import AppConfig


def as_mock(target: object) -> MagicMock:
    """型付き属性に注入したモックを、モックとして参照するためのキャスト"""
    return cast(MagicMock, target)


def make_label(mocker: MockerFixture, text: str) -> ttk.Label:
    """`label['text']` で値を返すttk.Label相当のモック"""
    label = mocker.MagicMock()
    label.__getitem__.return_value = text
    return cast(ttk.Label, label)


@pytest.fixture
def app(app_config: AppConfig, mocker: MockerFixture) -> PDFProcessorApp:
    instance = PDFProcessorApp.__new__(PDFProcessorApp)
    instance.master = mocker.MagicMock()
    instance.config = app_config
    instance.status_text = mocker.MagicMock()
    instance.observer = None
    instance.is_watching = False
    return instance


# --- 起動時の配線（スモークテスト） ---


def test_init_wires_startup_sequence(
    app_config: AppConfig,
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """ウィジェット生成から監視開始までを、tkinterを実起動せずに一度通す"""
    mocker.patch('app.main_window.tk')
    ttk_module = mocker.patch('app.main_window.ttk')
    ttk_module.Frame.return_value.winfo_children.return_value = [mocker.MagicMock()]
    mocker.patch('app.main_window.AppConfig', return_value=app_config)
    observer_class = mocker.patch('app.main_window.Observer')
    process = mocker.patch('app.main_window.process_pdf')
    (Path(app_config.processing_dir) / 'existing.pdf').write_bytes(b'pdf')
    master = mocker.MagicMock()

    instance = PDFProcessorApp(master)

    master.title.assert_called_once_with(f'BarcodePDF v{__version__}')
    master.geometry.assert_called_once_with('600x500')
    process.assert_called_once()
    observer_class.return_value.start.assert_called_once()
    assert instance.is_watching is True
    assert master.protocol.call_args.args[0] == 'WM_DELETE_WINDOW'


# --- update_status（P2） ---


def test_update_status_appends_and_restores_disabled_state(
    app: PDFProcessorApp,
    mocker: MockerFixture,
) -> None:
    app.update_status('進捗メッセージ')

    as_mock(app.status_text).insert.assert_called_once_with(tk.END, '進捗メッセージ\n')
    assert [call.kwargs['state'] for call in as_mock(app.status_text).config.call_args_list] == [
        tk.NORMAL,
        tk.DISABLED,
    ]


# --- ensure_directories（P1） ---


def test_ensure_directories_reports_created_directories(
    app: PDFProcessorApp,
    tmp_path: Path,
) -> None:
    (tmp_path / 'done').rmdir()

    app.ensure_directories()

    assert Path(app.config.done_dir).is_dir()
    assert as_mock(app.status_text).insert.call_count == 1


def test_ensure_directories_is_silent_when_nothing_created(app: PDFProcessorApp) -> None:
    app.ensure_directories()

    as_mock(app.status_text).insert.assert_not_called()


# --- process_existing_pdfs（P1） ---


def test_process_existing_pdfs_processes_only_pdf_files(
    app: PDFProcessorApp,
    mocker: MockerFixture,
) -> None:
    processing_dir = Path(app.config.processing_dir)
    (processing_dir / 'a.pdf').write_bytes(b'pdf')
    (processing_dir / 'B.PDF').write_bytes(b'pdf')
    (processing_dir / 'note.txt').write_text('text', encoding='utf-8')
    process = mocker.patch('app.main_window.process_pdf')

    app.process_existing_pdfs()

    processed = sorted(Path(call.args[0]).name for call in process.call_args_list)
    assert processed == ['B.PDF', 'a.pdf']


def test_process_existing_pdfs_handles_empty_directory(
    app: PDFProcessorApp,
    mocker: MockerFixture,
) -> None:
    process = mocker.patch('app.main_window.process_pdf')

    app.process_existing_pdfs()

    process.assert_not_called()


def test_process_existing_pdfs_passes_status_callback(
    app: PDFProcessorApp,
    mocker: MockerFixture,
) -> None:
    (Path(app.config.processing_dir) / 'a.pdf').write_bytes(b'pdf')
    process = mocker.patch('app.main_window.process_pdf')

    app.process_existing_pdfs()

    assert process.call_args.args[1] is app.config
    assert process.call_args.args[2] == app.update_status


# --- 監視の開始と停止（P1） ---


def test_start_watching_schedules_observer(app: PDFProcessorApp, mocker: MockerFixture) -> None:
    observer_class = mocker.patch('app.main_window.Observer')

    app.start_watching()

    observer_class.return_value.schedule.assert_called_once()
    assert observer_class.return_value.schedule.call_args.args[1] == app.config.processing_dir
    assert observer_class.return_value.schedule.call_args.kwargs == {'recursive': False}
    observer_class.return_value.start.assert_called_once()
    assert app.is_watching is True


def test_start_watching_is_idempotent(app: PDFProcessorApp, mocker: MockerFixture) -> None:
    observer_class = mocker.patch('app.main_window.Observer')

    app.start_watching()
    app.start_watching()

    assert observer_class.call_count == 1


def test_stop_watching_stops_observer(app: PDFProcessorApp, mocker: MockerFixture) -> None:
    observer = mocker.MagicMock()
    app.observer = observer
    app.is_watching = True

    app.stop_watching()

    observer.stop.assert_called_once()
    observer.join.assert_called_once()
    assert app.is_watching is False


def test_stop_watching_does_nothing_without_observer(app: PDFProcessorApp) -> None:
    app.stop_watching()

    as_mock(app.status_text).insert.assert_not_called()


# --- save_config（P1） ---


@pytest.fixture
def app_with_labels(
    app: PDFProcessorApp,
    mocker: MockerFixture,
    tmp_path: Path,
) -> PDFProcessorApp:
    app.processing_dir_label = make_label(mocker, str(tmp_path / 'new_processing'))
    app.error_dir_label = make_label(mocker, str(tmp_path / 'new_error'))
    app.done_dir_label = make_label(mocker, str(tmp_path / 'new_done'))
    app.log_dir_label = make_label(mocker, str(tmp_path / 'new_log'))
    app.auto_open_var = mocker.MagicMock()
    app.auto_open_var.get.return_value = True
    return app


def test_save_config_persists_label_values(
    app_with_labels: PDFProcessorApp,
    mocker: MockerFixture,
    config_file: Path,
    tmp_path: Path,
) -> None:
    mocker.patch('app.main_window.messagebox')
    mocker.patch('app.main_window.setup_logging')

    app_with_labels.save_config()

    reloaded = AppConfig(config_file)
    assert reloaded.processing_dir == str(tmp_path / 'new_processing')
    assert reloaded.error_dir == str(tmp_path / 'new_error')
    assert reloaded.done_dir == str(tmp_path / 'new_done')
    assert reloaded.log_dir == str(tmp_path / 'new_log')
    assert reloaded.auto_open_error_folder is True


def test_save_config_creates_new_directories(
    app_with_labels: PDFProcessorApp,
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    mocker.patch('app.main_window.messagebox')
    mocker.patch('app.main_window.setup_logging')

    app_with_labels.save_config()

    assert (tmp_path / 'new_processing').is_dir()
    assert (tmp_path / 'new_done').is_dir()


def test_save_config_reinitializes_logging_and_notifies(
    app_with_labels: PDFProcessorApp,
    mocker: MockerFixture,
) -> None:
    messagebox = mocker.patch('app.main_window.messagebox')
    setup_logging = mocker.patch('app.main_window.setup_logging')

    app_with_labels.save_config()

    setup_logging.assert_called_once_with(app_with_labels.config.config)
    messagebox.showinfo.assert_called_once()


# --- 終了処理（P2） ---


def test_quit_app_stops_watching_and_quits(app: PDFProcessorApp, mocker: MockerFixture) -> None:
    observer = mocker.MagicMock()
    app.observer = observer
    app.is_watching = True

    app.quit_app()

    observer.stop.assert_called_once()
    as_mock(app.master).quit.assert_called_once()


@pytest.mark.parametrize('confirmed, expected_calls', [(True, 1), (False, 0)])
def test_on_closing_respects_confirmation(
    confirmed: bool,
    expected_calls: int,
    app: PDFProcessorApp,
    mocker: MockerFixture,
) -> None:
    mocker.patch('app.main_window.messagebox.askokcancel', return_value=confirmed)
    quit_app = mocker.patch.object(app, 'quit_app')

    app.on_closing()

    assert quit_app.call_count == expected_calls


# --- browse_directory（P2） ---


def test_browse_directory_updates_label(mocker: MockerFixture) -> None:
    mocker.patch('app.main_window.filedialog.askdirectory', return_value='C:/selected')
    label = mocker.MagicMock()

    PDFProcessorApp.browse_directory(label)

    label.config.assert_called_once_with(text='C:/selected')


def test_browse_directory_keeps_label_when_cancelled(mocker: MockerFixture) -> None:
    mocker.patch('app.main_window.filedialog.askdirectory', return_value='')
    label = mocker.MagicMock()

    PDFProcessorApp.browse_directory(label)

    label.config.assert_not_called()
