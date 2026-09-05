"""エラーフォルダに溜まったPDFの削除"""

from utils.config_manager import AppConfig
from utils.constants import LABEL_ERROR_PDF
from utils.file_cleanup import delete_files_older_than


def cleanup_error_pdfs(config: AppConfig) -> None:
    """エラーフォルダのPDFのうち、保存期間を過ぎたものを削除する"""
    delete_files_older_than(
        config.error_dir,
        config.log_retention_days,
        lambda filename: filename.lower().endswith('.pdf'),
        LABEL_ERROR_PDF,
    )
