"""バーコード読み取り結果に応じたPDFの振り分け"""

import logging
import os
import shutil
import subprocess
from collections.abc import Callable

from service.barcode_reader import read_barcode_from_pdf
from utils.config_manager import AppConfig
from utils.constants import (
    MSG_BARCODE_INVALID,
    MSG_BARCODE_NOT_FOUND,
    MSG_FILE_NOT_FOUND,
    MSG_MOVE_ERROR,
    MSG_MOVED_TO_ERROR,
    MSG_OPEN_ERROR_FOLDER_FAILED,
    MSG_OPEN_ERROR_FOLDER_UNSUPPORTED,
    MSG_PROCESS_DONE,
    MSG_PROCESS_ERROR,
    MSG_PROCESSING_START,
    TRACE_FORMAT,
    TRACE_RESULT_ERROR,
    TRACE_RESULT_INVALID_BARCODE,
    TRACE_RESULT_NO_BARCODE,
    TRACE_RESULT_SUCCESS,
)

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]

# Windowsのファイル名に使用できない文字とデバイス名
INVALID_FILENAME_CHARS = '<>:"/\\|?*'
RESERVED_FILENAMES = frozenset(
    ['CON', 'PRN', 'AUX', 'NUL']
    + [f'COM{i}' for i in range(1, 10)]
    + [f'LPT{i}' for i in range(1, 10)]
)
MAX_BARCODE_LENGTH = 200


def is_valid_barcode(barcode: str) -> bool:
    """バーコード文字列をそのままファイル名に使えるかを判定する"""
    if not barcode or len(barcode) > MAX_BARCODE_LENGTH:
        return False
    if any(char in INVALID_FILENAME_CHARS or ord(char) < 32 for char in barcode):
        return False
    # デバイス名の予約は拡張子の有無に関わらず適用される（CON.foo も作成できない）
    if barcode.upper().split('.')[0] in RESERVED_FILENAMES:
        return False
    # 先頭・末尾の空白とピリオドはWindowsが除去するため別名になる（'..' もここで弾かれる）
    return barcode.strip(' .') == barcode


def log_trace(result: str, source: str, barcode: str | None, destination: str | None) -> None:
    """どのファイルをどこへ送ったかを1行で追跡できる形式で記録する"""
    logger.info(TRACE_FORMAT.format(
        result=result,
        source=source,
        barcode=barcode or '',
        destination=destination or '',
    ))


def open_error_folder(error_dir: str) -> None:
    try:
        if os.name == 'nt':
            os.startfile(error_dir)
        elif os.name == 'posix':
            subprocess.call(['open', error_dir])
        else:
            logger.warning(MSG_OPEN_ERROR_FOLDER_UNSUPPORTED.format(path=error_dir))
    except Exception as e:
        logger.error(MSG_OPEN_ERROR_FOLDER_FAILED.format(error=str(e)))


def _move_overwriting(source: str, destination: str) -> None:
    """同名ファイルがあれば上書きして移動する"""
    if os.path.exists(destination):
        os.remove(destination)
    shutil.move(source, destination)


def _move_to_done_dir(
    pdf_path: str,
    barcode_data: str,
    config: AppConfig,
    status_callback: StatusCallback,
) -> None:
    filename = os.path.basename(pdf_path)
    new_filename = f"{barcode_data}.pdf"
    done_path = os.path.join(config.done_dir, new_filename)

    _move_overwriting(pdf_path, done_path)

    message = MSG_PROCESS_DONE.format(source=filename, destination=new_filename)
    logger.info(message)
    status_callback(message)
    log_trace(TRACE_RESULT_SUCCESS, pdf_path, barcode_data, done_path)


def _move_to_error_dir(
    pdf_path: str,
    config: AppConfig,
    status_callback: StatusCallback,
    result: str,
    barcode: str | None = None,
) -> None:
    filename = os.path.basename(pdf_path)
    error_path = os.path.join(config.error_dir, filename)

    _move_overwriting(pdf_path, error_path)

    message = MSG_MOVED_TO_ERROR.format(filename=filename)
    logger.info(message)
    status_callback(message)
    log_trace(result, pdf_path, barcode, error_path)

    if config.auto_open_error_folder:
        open_error_folder(config.error_dir)


def _handle_failed_file(pdf_path: str, config: AppConfig, status_callback: StatusCallback) -> None:
    if not os.path.exists(pdf_path):
        log_trace(TRACE_RESULT_ERROR, pdf_path, None, None)
        return

    try:
        _move_to_error_dir(pdf_path, config, status_callback, TRACE_RESULT_ERROR)
    except Exception as move_error:
        message = MSG_MOVE_ERROR.format(error=str(move_error))
        logger.error(message, exc_info=True)
        status_callback(message)
        log_trace(TRACE_RESULT_ERROR, pdf_path, None, None)


def process_pdf(pdf_path: str, config: AppConfig, status_callback: StatusCallback) -> None:
    try:
        if not os.path.exists(pdf_path):
            message = MSG_FILE_NOT_FOUND.format(path=pdf_path)
            logger.warning(message)
            status_callback(message)
            return

        logger.info(MSG_PROCESSING_START.format(path=pdf_path))
        barcode_data = read_barcode_from_pdf(pdf_path, config)

        if barcode_data and is_valid_barcode(barcode_data):
            _move_to_done_dir(pdf_path, barcode_data, config, status_callback)
            return

        filename = os.path.basename(pdf_path)
        if barcode_data:
            result = TRACE_RESULT_INVALID_BARCODE
            message = MSG_BARCODE_INVALID.format(filename=filename, barcode=barcode_data)
        else:
            result = TRACE_RESULT_NO_BARCODE
            message = MSG_BARCODE_NOT_FOUND.format(filename=filename)

        logger.warning(message)
        status_callback(message)
        _move_to_error_dir(pdf_path, config, status_callback, result, barcode_data)

    except Exception as e:
        message = MSG_PROCESS_ERROR.format(filename=os.path.basename(pdf_path), error=str(e))
        logger.error(message, exc_info=True)
        status_callback(message)
        _handle_failed_file(pdf_path, config, status_callback)

