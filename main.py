import tkinter as tk

from app.main_window import PDFProcessorApp
from service.error_pdf_cleanup import cleanup_error_pdfs
from utils.config_manager import AppConfig
from utils.log_rotation import setup_logging


def main() -> None:
    setup_logging()
    # ログ設定の副作用ではなく、起動時に明示的にエラーPDFを片付ける
    cleanup_error_pdfs(AppConfig())
    root = tk.Tk()
    PDFProcessorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
