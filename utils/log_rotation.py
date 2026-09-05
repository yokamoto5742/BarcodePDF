import configparser
import logging
import os
import re
from logging.handlers import TimedRotatingFileHandler

from utils.config_manager import load_config
from utils.constants import (
    LABEL_LOG_FILE,
    MSG_LOG_DIR_PERMISSION_ERROR,
    MSG_LOG_INITIALIZED,
    MSG_LOG_LEVEL_INVALID,
    MSG_LOG_SETUP_ERROR,
)
from utils.file_cleanup import delete_files_older_than


def _resolve_log_directory(config: configparser.ConfigParser) -> str:
    """LOGGING/log_directory を絶対パスで返す（相対指定はプロジェクトルート基準）"""
    directory = config.get('LOGGING', 'log_directory', fallback='logs')
    if os.path.isabs(directory):
        return directory
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), directory)


def setup_logging(config: configparser.ConfigParser | None = None) -> None:
    if config is None:
        config = load_config()

    try:
        log_directory = _resolve_log_directory(config)
        log_retention_days = config.getint('LOGGING', 'log_retention_days', fallback=7)
        project_name = config.get('LOGGING', 'project_name', fallback='BarcodePDF')
        log_level = config.get('LOGGING', 'log_level', fallback='INFO')

        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        log_file = os.path.join(log_directory, f'{project_name}.log')

        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when='midnight',
            backupCount=log_retention_days,
            encoding='utf-8'
        )
        file_handler.suffix = "%Y-%m-%d.log"

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        root_logger = logging.getLogger()

        # 設定変更後の再初期化でハンドラが重複しないよう既存分を除去する
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

        # logging には int 以外の属性もあるため、レベル値であることまで確認する
        level = getattr(logging, log_level.upper(), None)
        if isinstance(level, int):
            root_logger.setLevel(level)
        else:
            root_logger.setLevel(logging.INFO)
            logging.warning(MSG_LOG_LEVEL_INVALID.format(level=log_level))

        root_logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.WARNING)
        root_logger.addHandler(console_handler)

        cleanup_old_logs(log_directory, log_retention_days, project_name)

        logging.info(MSG_LOG_INITIALIZED.format(path=log_file))

    except PermissionError as e:
        raise PermissionError(MSG_LOG_DIR_PERMISSION_ERROR.format(error=e)) from e
    except Exception as e:
        raise Exception(MSG_LOG_SETUP_ERROR.format(error=e)) from e


def cleanup_old_logs(log_directory: str, retention_days: int, project_name: str) -> None:
    """ローテーション済みログのうち、保存期間を過ぎたものを削除する"""
    rotated_log_pattern = re.compile(
        rf'{re.escape(project_name)}\.log\.\d{{4}}-\d{{2}}-\d{{2}}\.log$'
    )
    delete_files_older_than(
        log_directory,
        retention_days,
        lambda filename: bool(rotated_log_pattern.match(filename)),
        LABEL_LOG_FILE,
    )
