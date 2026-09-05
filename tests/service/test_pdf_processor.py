"""P0: PDFの振り分け"""

import logging
import os
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from service.pdf_processor import (
    MAX_BARCODE_LENGTH,
    is_valid_barcode,
    log_trace,
    open_error_folder,
    process_pdf,
)
from utils.config_manager import AppConfig
from utils.constants import (
    TRACE_RESULT_ERROR,
    TRACE_RESULT_INVALID_BARCODE,
    TRACE_RESULT_NO_BARCODE,
    TRACE_RESULT_SUCCESS,
)


def patch_barcode(mocker: MockerFixture, value: str | None) -> None:
    mocker.patch('service.pdf_processor.read_barcode_from_pdf', return_value=value)


def trace_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.message for record in caplog.records if record.message.startswith('TRACE ')]


# --- is_valid_barcode（P0: ファイル名としての安全性判定） ---


@pytest.mark.parametrize('barcode', [
    '12345',
    'ABC-123',
    '患者001',
    'a' * MAX_BARCODE_LENGTH,
    'A..B',
    'CONTAINER',
])
def test_is_valid_barcode_accepts_usable_names(barcode: str) -> None:
    assert is_valid_barcode(barcode) is True


@pytest.mark.parametrize('barcode', [
    '',
    '..',
    '../evil',
    'A/B',
    'A\\B',
    'C:evil',
    'a<b',
    'a>b',
    'a"b',
    'a|b',
    'a?b',
    'a*b',
    'a\x00b',
    'a\nb',
    ' leading',
    'trailing ',
    'trailing.',
    'CON',
    'nul',
    'COM1',
    'LPT9',
    'a' * (MAX_BARCODE_LENGTH + 1),
])
def test_is_valid_barcode_rejects_unsafe_names(barcode: str) -> None:
    assert is_valid_barcode(barcode) is False


# --- process_pdf 正常系 ---


def test_process_pdf_moves_to_done_dir_with_barcode_name(
    mocker: MockerFixture,
    app_config: AppConfig,
    pdf_in_target: Path,
    status_messages: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    patch_barcode(mocker, 'ABC123')

    process_pdf(str(pdf_in_target), app_config, status_messages.append)

    assert (Path(app_config.done_dir) / 'ABC123.pdf').exists()
    assert not pdf_in_target.exists()
    assert any('ABC123.pdf' in message for message in status_messages)
    assert f'result={TRACE_RESULT_SUCCESS}' in trace_lines(caplog)[0]


def test_process_pdf_overwrites_existing_done_file(
    mocker: MockerFixture,
    app_config: AppConfig,
    pdf_in_target: Path,
    status_messages: list[str],
) -> None:
    existing = Path(app_config.done_dir) / 'ABC123.pdf'
    existing.write_bytes(b'old content')
    patch_barcode(mocker, 'ABC123')

    process_pdf(str(pdf_in_target), app_config, status_messages.append)

    assert existing.read_bytes() == b'%PDF-1.7 dummy'
    assert not pdf_in_target.exists()


# --- process_pdf 異常系 ---


def test_process_pdf_moves_to_error_dir_when_barcode_not_found(
    mocker: MockerFixture,
    app_config: AppConfig,
    pdf_in_target: Path,
    status_messages: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    patch_barcode(mocker, None)

    process_pdf(str(pdf_in_target), app_config, status_messages.append)

    assert (Path(app_config.error_dir) / 'input.pdf').exists()
    assert not pdf_in_target.exists()
    assert f'result={TRACE_RESULT_NO_BARCODE}' in trace_lines(caplog)[0]


@pytest.mark.parametrize('barcode', ['../evil', 'A/B', 'CON', 'trailing.', 'a' * 300])
def test_process_pdf_moves_to_error_dir_when_barcode_is_invalid(
    barcode: str,
    mocker: MockerFixture,
    app_config: AppConfig,
    pdf_in_target: Path,
    status_messages: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    patch_barcode(mocker, barcode)

    process_pdf(str(pdf_in_target), app_config, status_messages.append)

    assert (Path(app_config.error_dir) / 'input.pdf').exists()
    assert not pdf_in_target.exists()
    # 完了フォルダ配下にも、その外にもファイルを作らない
    assert list(Path(app_config.done_dir).iterdir()) == []
    assert f'result={TRACE_RESULT_INVALID_BARCODE}' in trace_lines(caplog)[0]
    assert f'barcode={barcode}' in trace_lines(caplog)[0]
    assert any(barcode in message for message in status_messages)


def test_process_pdf_overwrites_existing_error_file(
    mocker: MockerFixture,
    app_config: AppConfig,
    pdf_in_target: Path,
    status_messages: list[str],
) -> None:
    existing = Path(app_config.error_dir) / 'input.pdf'
    existing.write_bytes(b'old content')
    patch_barcode(mocker, None)

    process_pdf(str(pdf_in_target), app_config, status_messages.append)

    assert existing.read_bytes() == b'%PDF-1.7 dummy'


def test_process_pdf_returns_early_when_file_missing(
    mocker: MockerFixture,
    app_config: AppConfig,
    status_messages: list[str],
) -> None:
    read_barcode = mocker.patch('service.pdf_processor.read_barcode_from_pdf')
    missing = os.path.join(app_config.target_dir, 'missing.pdf')

    process_pdf(missing, app_config, status_messages.append)

    read_barcode.assert_not_called()
    assert any('missing.pdf' in message for message in status_messages)


def test_process_pdf_moves_to_error_dir_when_read_raises(
    mocker: MockerFixture,
    app_config: AppConfig,
    pdf_in_target: Path,
    status_messages: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    mocker.patch(
        'service.pdf_processor.read_barcode_from_pdf',
        side_effect=RuntimeError('PDF破損'),
    )

    process_pdf(str(pdf_in_target), app_config, status_messages.append)

    assert (Path(app_config.error_dir) / 'input.pdf').exists()
    assert any('PDF破損' in message for message in status_messages)
    assert f'result={TRACE_RESULT_ERROR}' in trace_lines(caplog)[-1]


def test_process_pdf_swallows_error_when_move_also_fails(
    mocker: MockerFixture,
    app_config: AppConfig,
    pdf_in_target: Path,
    status_messages: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    patch_barcode(mocker, 'ABC123')
    mocker.patch('service.pdf_processor.shutil.move', side_effect=PermissionError('使用中'))

    process_pdf(str(pdf_in_target), app_config, status_messages.append)

    assert pdf_in_target.exists()
    assert any('使用中' in message for message in status_messages)
    assert f'result={TRACE_RESULT_ERROR}' in trace_lines(caplog)[-1]


def test_process_pdf_logs_trace_without_destination_when_file_vanished(
    mocker: MockerFixture,
    app_config: AppConfig,
    pdf_in_target: Path,
    status_messages: list[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """移動成功後の後処理で失敗した場合、退避対象のファイルは既に存在しない"""
    caplog.set_level(logging.INFO)
    patch_barcode(mocker, 'ABC123')
    mocker.patch('service.pdf_processor.log_trace', side_effect=[RuntimeError('後処理失敗'), None])

    process_pdf(str(pdf_in_target), app_config, status_messages.append)

    assert (Path(app_config.done_dir) / 'ABC123.pdf').exists()
    assert not pdf_in_target.exists()
    assert any('後処理失敗' in message for message in status_messages)


@pytest.mark.parametrize('auto_open, expected_calls', [(True, 1), (False, 0)])
def test_process_pdf_opens_error_folder_by_config(
    auto_open: bool,
    expected_calls: int,
    mocker: MockerFixture,
    app_config: AppConfig,
    pdf_in_target: Path,
    status_messages: list[str],
) -> None:
    app_config.auto_open_error_folder = auto_open
    patch_barcode(mocker, None)
    open_folder = mocker.patch('service.pdf_processor.open_error_folder')

    process_pdf(str(pdf_in_target), app_config, status_messages.append)

    assert open_folder.call_count == expected_calls


# --- log_trace ---


def test_log_trace_replaces_none_with_empty_string(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    log_trace(TRACE_RESULT_ERROR, 'C:/in/a.pdf', None, None)

    assert caplog.records[0].message == (
        f'TRACE result={TRACE_RESULT_ERROR} src=C:/in/a.pdf barcode= dst='
    )


# --- open_error_folder（P2） ---


def test_open_error_folder_on_windows(mocker: MockerFixture) -> None:
    mocker.patch('service.pdf_processor.os.name', 'nt')
    startfile = mocker.patch('service.pdf_processor.os.startfile', create=True)

    open_error_folder('C:/error')

    startfile.assert_called_once_with('C:/error')


def test_open_error_folder_on_posix(mocker: MockerFixture) -> None:
    mocker.patch('service.pdf_processor.os.name', 'posix')
    call = mocker.patch('service.pdf_processor.subprocess.call')

    open_error_folder('/tmp/error')

    call.assert_called_once_with(['open', '/tmp/error'])


def test_open_error_folder_on_unsupported_os(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mocker.patch('service.pdf_processor.os.name', 'java')

    open_error_folder('/error')

    assert caplog.records[0].levelno == logging.WARNING


def test_open_error_folder_logs_error_on_failure(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mocker.patch('service.pdf_processor.os.name', 'nt')
    mocker.patch('service.pdf_processor.os.startfile', create=True, side_effect=OSError('失敗'))

    open_error_folder('C:/error')

    assert caplog.records[0].levelno == logging.ERROR
    assert '失敗' in caplog.records[0].message

