"""target_dir に入ったファイルを processing_dir へ自動的に移動する常駐スクリプト

実行方法:
    .venv\\Scripts\\python.exe -m scripts.file_move
"""

import logging
import os
import shutil
import sys
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config_manager import AppConfig
from utils.constants import (
    MSG_DIRECTORY_CREATED,
    MSG_FILE_MOVE_ERROR,
    MSG_FILE_MOVED,
    MSG_WATCH_STARTED,
    MSG_WATCH_STOPPED,
)

logger = logging.getLogger(__name__)

# 書き込み途中のファイルを移動すると失敗するため、検出後に待機する秒数
FILE_WRITE_WAIT_SECONDS = 1
# 監視ループが停止を確認する間隔
POLL_INTERVAL_SECONDS = 1


def move_file(file_path: str, processing_dir: str) -> None:
    """1ファイルを processing_dir へ移動する。同名ファイルは上書きする"""
    filename = os.path.basename(file_path)
    destination = os.path.join(processing_dir, filename)

    try:
        if os.path.exists(destination):
            os.remove(destination)
        shutil.move(file_path, destination)
        logger.info(MSG_FILE_MOVED.format(source=file_path, destination=destination))
    except OSError as e:
        logger.error(MSG_FILE_MOVE_ERROR.format(filename=filename, error=str(e)))


def move_existing_files(target_dir: str, processing_dir: str) -> None:
    """監視開始前から target_dir にあるファイルを移動する"""
    for entry in os.scandir(target_dir):
        if entry.is_file():
            move_file(entry.path, processing_dir)


class FileMoveHandler(FileSystemEventHandler):
    def __init__(self, processing_dir: str) -> None:
        self.processing_dir = processing_dir

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        time.sleep(FILE_WRITE_WAIT_SECONDS)
        move_file(str(event.src_path), self.processing_dir)


def watch(target_dir: str, processing_dir: str) -> None:
    observer = Observer()
    observer.schedule(FileMoveHandler(processing_dir), target_dir, recursive=False)
    observer.start()
    logger.info(MSG_WATCH_STARTED.format(directory=target_dir))

    try:
        while True:
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        observer.stop()
        logger.info(MSG_WATCH_STOPPED)
    observer.join()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    config = AppConfig()
    for directory in config.ensure_directories():
        logger.info(MSG_DIRECTORY_CREATED.format(directory=directory))

    move_existing_files(config.target_dir, config.processing_dir)
    watch(config.target_dir, config.processing_dir)


if __name__ == "__main__":
    main()
