"""バーコードPDF処理アプリのGUI"""

import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import cast

from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from app import __version__
from service.pdf_processor import PDFHandler, process_pdf
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
    LABEL_PROCESSING_DIR,
    LABEL_STATUS,
    MSG_APP_QUIT,
    MSG_CONFIG_UPDATED,
    MSG_EXISTING_PDF_DONE,
    MSG_EXISTING_PDF_START,
    MSG_WATCH_STARTED,
    MSG_WATCH_STOPPED,
    VERSION_LABEL,
)
from utils.log_rotation import setup_logging

logger = logging.getLogger(__name__)


class PDFProcessorApp:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title(APP_TITLE)
        self.config = AppConfig()
        self.master.geometry(f"{self.config.ui_width}x{self.config.ui_height}")

        self.create_widgets()
        self.observer: BaseObserver | None = None
        self.is_watching = False

        self.process_existing_pdfs()
        self.start_watching()

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self) -> None:
        self.frame = ttk.Frame(self.master, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        ttk.Label(
            self.frame, text=VERSION_LABEL.format(version=__version__), font=("", 10, "bold")
        ).grid(column=0, row=0, columnspan=3, sticky=tk.W)

        self.processing_dir_label = self._create_directory_row(LABEL_PROCESSING_DIR, self.config.processing_dir, 1)
        self.error_dir_label = self._create_directory_row(LABEL_ERROR_DIR, self.config.error_dir, 2)
        self.done_dir_label = self._create_directory_row(LABEL_DONE_DIR, self.config.done_dir, 3)
        self.log_dir_label = self._create_directory_row(LABEL_LOG_DIR, self.config.log_dir, 4)

        self.auto_open_var = tk.BooleanVar(value=self.config.auto_open_error_folder)
        ttk.Checkbutton(
            self.frame,
            text=CHECKBOX_AUTO_OPEN_ERROR_FOLDER,
            variable=self.auto_open_var,
        ).grid(column=0, row=5, columnspan=2, sticky=tk.W)

        ttk.Button(self.frame, text=BUTTON_SAVE_CONFIG, command=self.save_config).grid(column=2, row=5, sticky=tk.E)
        ttk.Button(self.frame, text=BUTTON_CLOSE, command=self.quit_app).grid(column=2, row=6, sticky=tk.E)

        ttk.Label(self.frame, text=LABEL_STATUS).grid(column=0, row=7, sticky=tk.W)

        self.status_text = tk.Text(self.frame, height=10, width=70, wrap=tk.WORD)
        self.status_text.grid(column=0, row=8, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.status_text.config(state=tk.DISABLED)

        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(column=3, row=8, sticky=(tk.N, tk.S))
        self.status_text['yscrollcommand'] = scrollbar.set

        for child in self.frame.winfo_children():
            cast(tk.Widget, child).grid_configure(padx=5, pady=5)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(8, weight=1)

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
        self.config.processing_dir = str(self.processing_dir_label['text'])
        self.config.error_dir = str(self.error_dir_label['text'])
        self.config.done_dir = str(self.done_dir_label['text'])
        self.config.log_dir = str(self.log_dir_label['text'])
        self.config.auto_open_error_folder = self.auto_open_var.get()
        self.config.save()

        setup_logging(self.config.config)
        messagebox.showinfo(DIALOG_SAVE_CONFIG_TITLE, DIALOG_SAVE_CONFIG_MESSAGE)
        logger.info(MSG_CONFIG_UPDATED)

    def process_existing_pdfs(self) -> None:
        logger.info(MSG_EXISTING_PDF_START)
        self.update_status(MSG_EXISTING_PDF_START)

        for filename in os.listdir(self.config.processing_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.config.processing_dir, filename)
                process_pdf(pdf_path, self.config, self.update_status)

        logger.info(MSG_EXISTING_PDF_DONE)
        self.update_status(MSG_EXISTING_PDF_DONE)

    def start_watching(self) -> None:
        if self.is_watching:
            return

        self.observer = Observer()
        event_handler = PDFHandler(self.config, self.update_status)
        self.observer.schedule(event_handler, self.config.processing_dir, recursive=False)
        self.observer.start()
        self.is_watching = True

        message = MSG_WATCH_STARTED.format(directory=self.config.processing_dir)
        logger.info(message)
        self.update_status(message)

    def stop_watching(self) -> None:
        if not self.observer:
            return

        self.observer.stop()
        self.observer.join()
        self.is_watching = False

        logger.info(MSG_WATCH_STOPPED)
        self.update_status(MSG_WATCH_STOPPED)

    def update_status(self, message: str) -> None:
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
