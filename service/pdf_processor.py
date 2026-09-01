"""バーコード読み取り結果に応じたPDFの振り分けとフォルダ監視"""

import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler

from service.barcode_reader import read_barcode_from_pdf
from utils.config_manager import AppConfig
from utils.constants import (
    MSG_BARCODE_NOT_FOUND,
    MSG_FILE_NOT_FOUND,
    MSG_MOVE_ERROR,
    MSG_MOVED_TO_ERROR,
    MSG_OPEN_ERROR_FOLDER_FAILED,
    MSG_OPEN_ERROR_FOLDER_UNSUPPORTED,
    MSG_PDF_DETECTED,
    MSG_PROCESS_DONE,
    MSG_PROCESS_ERROR,
    MSG_PROCESSING_START,
    TRACE_FORMAT,
    TRACE_RESULT_ERROR,
    TRACE_RESULT_NO_BARCODE,
    TRACE_RESULT_SUCCESS,
)

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]

# 書き込み途中のファイルを読むと失敗するため、検出後に待機する秒数
FILE_WRITE_WAIT_SECONDS = 1


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


def _move_to_done_dir(
    pdf_path: str,
    barcode_data: str,
    config: AppConfig,
    status_callback: StatusCallback,
) -> None:
    new_filename = f"{barcode_data}.pdf"
    done_path = os.path.join(config.done_dir, new_filename)
    shutil.move(pdf_path, done_path)

    message = MSG_PROCESS_DONE.format(source=os.path.basename(pdf_path), destination=new_filename)
    logger.info(message)
    status_callback(message)
    log_trace(TRACE_RESULT_SUCCESS, pdf_path, barcode_data, done_path)


def _move_to_error_dir(
    pdf_path: str,
    config: AppConfig,
    status_callback: StatusCallback,
    result: str,
) -> None:
    error_path = os.path.join(config.error_dir, os.path.basename(pdf_path))
    shutil.move(pdf_path, error_path)

    message = MSG_MOVED_TO_ERROR.format(filename=os.path.basename(pdf_path))
    logger.info(message)
    status_callback(message)
    log_trace(result, pdf_path, None, error_path)

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
        barcode_data = read_barcode_from_pdf(pdf_path)

        if barcode_data:
            _move_to_done_dir(pdf_path, barcode_data, config, status_callback)
        else:
            message = MSG_BARCODE_NOT_FOUND.format(filename=os.path.basename(pdf_path))
            logger.warning(message)
            status_callback(message)
            _move_to_error_dir(pdf_path, config, status_callback, TRACE_RESULT_NO_BARCODE)

    except Exception as e:
        message = MSG_PROCESS_ERROR.format(filename=os.path.basename(pdf_path), error=str(e))
        logger.error(message, exc_info=True)
        status_callback(message)
        _handle_failed_file(pdf_path, config, status_callback)


class PDFHandler(FileSystemEventHandler):
    def __init__(self, config: AppConfig, status_callback: StatusCallback) -> None:
        self.config = config
        self.status_callback = status_callback

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        src_path = str(event.src_path)
        if not src_path.lower().endswith('.pdf'):
            return

        message = MSG_PDF_DETECTED.format(path=src_path)
        logger.info(message)
        self.status_callback(message)

        time.sleep(FILE_WRITE_WAIT_SECONDS)
        process_pdf(src_path, self.config, self.status_callback)
