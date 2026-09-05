"""target_dir に入ったPDFを監視してバーコード処理へ渡すフォルダ監視

スキャナーによってはスキャン直後にファイル名を変更するため、ファイルイベントに依存すると
移動元が消えて取りこぼす。サイズと更新日時が前回走査から変化していないことを確認してから
処理することで、リネームや書き込み途中のファイルを避ける。
"""

import logging
import os
import threading
from collections.abc import Callable

from service.pdf_processor import process_pdf
from utils.config_manager import AppConfig
from utils.constants import MSG_SCAN_ERROR

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


class PdfWatcher:
    """target_dir を一定間隔で走査し、書き込みが完了したPDFを処理する"""

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
        self._handled: set[str] = set()

    def start(self) -> None:
        if self._thread:
            return

        # stop 後に再度 start しても _run が即終了しないよう落としておく
        self._stop_event.clear()
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
        """target_dir を1回走査し、前回と同じ状態のPDFを処理する

        処理を試みたファイルも記録する。移動に失敗して取込フォルダに残ったファイルを、
        内容が変わらないまま繰り返し処理しないため。
        """
        current_signatures: dict[str, FileSignature] = {}

        for entry in self._scan_pdf_entries():
            signature = _file_signature(entry.path)
            if signature is None:
                continue

            previous = self._signatures.get(entry.path)
            current_signatures[entry.path] = signature

            if previous != signature:
                # 内容が変わった＝別のファイルが置かれたので処理対象に戻す
                self._handled.discard(entry.path)
            elif entry.path not in self._handled:
                # 前回と同じサイズ・更新日時になった初回だけ、書き込み完了とみなす
                self._handled.add(entry.path)
                process_pdf(entry.path, self.config, self.status_callback)

        self._signatures = current_signatures
        # 消えたファイルの記録は残さない（set が際限なく育つのを防ぐ）
        self._handled &= current_signatures.keys()

    def _scan_pdf_entries(self) -> list[os.DirEntry[str]]:
        try:
            return [
                entry for entry in os.scandir(self.config.target_dir)
                if entry.is_file() and entry.name.lower().endswith('.pdf')
            ]
        except OSError as e:
            logger.error(MSG_SCAN_ERROR.format(directory=self.config.target_dir, error=str(e)))
            return []
