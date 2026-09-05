"""保存期間を過ぎたファイルの削除"""

import logging
import os
from collections.abc import Callable
from datetime import datetime, timedelta

from utils.constants import MSG_DELETE_FAILED, MSG_DELETE_FILE, MSG_DELETE_SUMMARY


def delete_files_older_than(
    directory: str,
    retention_days: int,
    is_target: Callable[[str], bool],
    label: str,
) -> None:
    """directory 直下の対象ファイルのうち、保存期間を過ぎたものを削除する"""
    if not os.path.isdir(directory):
        return

    threshold = datetime.now() - timedelta(days=retention_days)
    deleted_count = 0

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not is_target(filename) or not os.path.isfile(file_path):
            continue

        try:
            if datetime.fromtimestamp(os.path.getmtime(file_path)) <= threshold:
                os.remove(file_path)
                logging.info(MSG_DELETE_FILE.format(label=label, filename=filename))
                deleted_count += 1
        except OSError as e:
            logging.error(MSG_DELETE_FAILED.format(label=label, filename=filename, error=str(e)))

    if deleted_count > 0:
        logging.info(MSG_DELETE_SUMMARY.format(count=deleted_count, label=label))
