"""バーコードPDF処理アプリのGUI"""

import logging
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import cast

from app import __version__
from service.pdf_watcher import PdfWatcher
from utils.config_manager import AppConfig
from utils.constants import (
    APP_TITLE,
    BUTTON_BROWSE,
    BUTTON_CLOSE,
    BUTTON_SAVE_CONFIG,
    CHECKBOX_AUTO_OPEN_ERROR_FOLDER,
    DIALOG_QUIT_MESSAGE,
    DIALOG_QUIT_TITLE,
    DIALOG_SAVE_CONFIG_MESSAGE,
    DIALOG_SAVE_CONFIG_TITLE,
    LABEL_DONE_DIR,
    LABEL_ERROR_DIR,
    LABEL_LOG_DIR,
    LABEL_STATUS,
    LABEL_TARGET_DIR,
    MSG_APP_QUIT,
    MSG_CONFIG_UPDATED,
    MSG_DIRECTORY_CREATED,
    MSG_WATCH_STARTED,
    MSG_WATCH_STOPPED,
)
from utils.log_rotation import setup_logging

logger = logging.getLogger(__name__)

# ステータスキューを取り出す間隔（ミリ秒）
STATUS_POLL_MS = 200


class PDFProcessorApp:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title(f"{APP_TITLE} v{__version__}")
        self.config = AppConfig()
        self.master.geometry(f"{self.config.ui_width}x{self.config.ui_height}")

        self._status_queue: queue.Queue[str] = queue.Queue()
        self.create_widgets()
        self._drain_status_queue()
        self.watcher = PdfWatcher(self.config, self.update_status)

        self.ensure_directories()
        self.start_watching()

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self) -> None:
        self.frame = ttk.Frame(self.master, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        self.target_dir_label = self._create_directory_row(LABEL_TARGET_DIR, self.config.target_dir, 0)
        self.error_dir_label = self._create_directory_row(LABEL_ERROR_DIR, self.config.error_dir, 1)
        self.done_dir_label = self._create_directory_row(LABEL_DONE_DIR, self.config.done_dir, 2)
        self.log_dir_label = self._create_directory_row(LABEL_LOG_DIR, self.config.log_dir, 3)

        self.auto_open_var = tk.BooleanVar(value=self.config.auto_open_error_folder)
        ttk.Checkbutton(
            self.frame,
            text=CHECKBOX_AUTO_OPEN_ERROR_FOLDER,
            variable=self.auto_open_var,
        ).grid(column=0, row=4, columnspan=2, sticky=tk.W)

        ttk.Button(self.frame, text=BUTTON_SAVE_CONFIG, command=self.save_config).grid(column=2, row=4, sticky=tk.E)
        ttk.Button(self.frame, text=BUTTON_CLOSE, command=self.quit_app).grid(column=2, row=5, sticky=tk.E)

        ttk.Label(self.frame, text=LABEL_STATUS).grid(column=0, row=6, sticky=tk.W)

        self.status_text = tk.Text(self.frame, height=10, width=70, wrap=tk.WORD)
        self.status_text.grid(column=0, row=7, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.status_text.config(state=tk.DISABLED)

        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(column=3, row=7, sticky=(tk.N, tk.S))
        self.status_text['yscrollcommand'] = scrollbar.set

        for child in self.frame.winfo_children():
            cast(tk.Widget, child).grid_configure(padx=5, pady=5)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(7, weight=1)

    def _create_directory_row(self, label_text: str, directory: str, row: int) -> ttk.Label:
        ttk.Label(self.frame, text=label_text).grid(column=0, row=row, sticky=tk.W)
        value_label = ttk.Label(self.frame, text=directory, width=50)
        value_label.grid(column=1, row=row, sticky=(tk.W, tk.E))
        ttk.Button(
            self.frame, text=BUTTON_BROWSE, command=lambda: self.browse_directory(value_label)
        ).grid(column=2, row=row)
        return value_label

    @staticmethod
    def browse_directory(target_label: ttk.Label) -> None:
        directory = filedialog.askdirectory()
        if directory:
            target_label.config(text=directory)

    def save_config(self) -> None:
        self.config.target_dir = str(self.target_dir_label['text'])
        self.config.error_dir = str(self.error_dir_label['text'])
        self.config.done_dir = str(self.done_dir_label['text'])
        self.config.log_dir = str(self.log_dir_label['text'])
        self.config.auto_open_error_folder = self.auto_open_var.get()
        self.config.save()
        self.ensure_directories()

        # 新しい取込フォルダを対象にするため監視を貼り直す
        self.stop_watching()
        self.start_watching()

        setup_logging(self.config.config)
        messagebox.showinfo(DIALOG_SAVE_CONFIG_TITLE, DIALOG_SAVE_CONFIG_MESSAGE)
        logger.info(MSG_CONFIG_UPDATED)

    def ensure_directories(self) -> None:
        for directory in self.config.ensure_directories():
            message = MSG_DIRECTORY_CREATED.format(directory=directory)
            logger.info(message)
            self.update_status(message)

    def start_watching(self) -> None:
        self.watcher.start()

        message = MSG_WATCH_STARTED.format(directory=self.config.target_dir)
        logger.info(message)
        self.update_status(message)

    def stop_watching(self) -> None:
        self.watcher.stop()

        logger.info(MSG_WATCH_STOPPED)
        self.update_status(MSG_WATCH_STOPPED)

    def update_status(self, message: str) -> None:
        """監視スレッドからも呼ばれるためウィジェットには直接触れずキューへ渡す"""
        self._status_queue.put(message)

    def _drain_status_queue(self) -> None:
        while True:
            try:
                message = self._status_queue.get_nowait()
            except queue.Empty:
                break
            self._append_status(message)

        self.master.after(STATUS_POLL_MS, self._drain_status_queue)

    def _append_status(self, message: str) -> None:
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

    def quit_app(self) -> None:
        self.stop_watching()
        logger.info(MSG_APP_QUIT)
        self.master.quit()

    def on_closing(self) -> None:
        if messagebox.askokcancel(DIALOG_QUIT_TITLE, DIALOG_QUIT_MESSAGE):
            self.quit_app()
