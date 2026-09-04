"""target_dir に入ったPDFを processing_dir へ移動するフォルダ監視

スキャナーによってはスキャン直後にファイル名を変更するため、ファイルイベントに依存すると
移動元が消えて取りこぼす。サイズと更新日時が前回走査から変化していないことを確認してから
移動することで、リネームや書き込み途中のファイルを避ける。
"""

import logging
import os
import shutil
import threading
from collections.abc import Callable

from utils.config_manager import AppConfig
from utils.constants import MSG_FILE_MOVE_ERROR, MSG_FILE_MOVED

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]

POLL_INTERVAL_SECONDS = 2.0

FileSignature = tuple[int, float]


def _file_signature(path: str) -> FileSignature | None:
    """ファイルのサイズと更新日時を返す。読めない場合は None"""
    try:
        stat_result = os.stat(path)
    except OSError:
        return None
    return stat_result.st_size, stat_result.st_mtime


class TargetDirWatcher:
    """target_dir を一定間隔で走査し、書き込みが完了したPDFを processing_dir へ移動する"""

    def __init__(
        self,
        config: AppConfig,
        status_callback: StatusCallback,
        poll_interval: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        self.config = config
        self.status_callback = status_callback
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._signatures: dict[str, FileSignature] = {}
        self._failed_paths: set[str] = set()

    def start(self) -> None:
        if self._thread:
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._thread:
            return

        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self.poll_interval):
            self.scan_once()

    def scan_once(self) -> None:
        """target_dir を1回走査し、前回と同じ状態のPDFを移動する"""
        stable_signatures: dict[str, FileSignature] = {}

        for entry in self._scan_pdf_entries():
            signature = _file_signature(entry.path)
            if signature is None:
                continue

            # 前回と同じサイズ・更新日時なら書き込みが完了している
            if self._signatures.get(entry.path) == signature and self._move_file(entry.path):
                continue

            stable_signatures[entry.path] = signature

        self._signatures = stable_signatures

    def _scan_pdf_entries(self) -> list[os.DirEntry[str]]:
        try:
            return [
                entry for entry in os.scandir(self.config.target_dir)
                if entry.is_file() and entry.name.lower().endswith('.pdf')
            ]
        except OSError as e:
            logger.error(MSG_FILE_MOVE_ERROR.format(filename=self.config.target_dir, error=str(e)))
            return []

    def _move_file(self, file_path: str) -> bool:
        """1ファイルを processing_dir へ移動する。同名ファイルは上書きする"""
        filename = os.path.basename(file_path)
        destination = os.path.join(self.config.processing_dir, filename)

        try:
            if os.path.exists(destination):
                os.remove(destination)
            shutil.move(file_path, destination)
        except OSError as e:
            # 移動できない間は走査のたびに再試行するため、ログは最初の1回だけ出す
            if file_path not in self._failed_paths:
                self._failed_paths.add(file_path)
                logger.error(MSG_FILE_MOVE_ERROR.format(filename=filename, error=str(e)))
            return False

        self._failed_paths.discard(file_path)
        message = MSG_FILE_MOVED.format(source=file_path, destination=destination)
        logger.info(message)
        self.status_callback(message)
        return True
