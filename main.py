import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pyzbar.pyzbar import decode
import fitz
import shutil
import configparser
import time

from pyzbar.wrapper import ZBarSymbol
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import subprocess

VERSION = "1.0.6"
LAST_UPDATED = "2024/08/12"


class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.processing_dir = self.config['Directories']['processing_dir']
        self.error_dir = self.config['Directories']['error_dir']
        self.done_dir = self.config['Directories']['done_dir']
        self.log_dir = self.config['Directories'].get('log_dir', 'log')
        self.ui_width = self.config['UI'].getint('width', 600)
        self.ui_height = self.config['UI'].getint('height', 500)
        self.auto_open_error_folder = self.config['Options'].getboolean('auto_open_error_folder', True)
        self.log_retention_days = self.config['Logging'].getint('retention_days', 14)
        self.start_minimized = self.config['Options'].getboolean('start_minimized', True)

    def save(self):
        self.config['Options'] = {
            'auto_open_error_folder': str(self.auto_open_error_folder),
            'start_minimized': str(self.start_minimized)
        }
        self.config['Logging'] = {
            'retention_days': str(self.log_retention_days)
        }
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)


def setup_logger(config):
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    log_file = os.path.join(config.log_dir, 'BarcodePDF.log')
    handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=config.log_retention_days)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger('BarcodePDF')
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def extract_images_from_pdf(pdf_path):
    images = []
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)
        # 埋め込み画像の抽出と処理
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            gray_image = image.convert('L')
            enhancer = ImageEnhance.Contrast(gray_image)
            enhanced_image = enhancer.enhance(2.0)
            images.append(enhanced_image)

        # ページ全体を画像として抽出し処理する
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        gray_img = img.convert('L')
        enhancer = ImageEnhance.Contrast(gray_img)
        enhanced_img = enhancer.enhance(2.0)
        images.append(enhanced_img)

    pdf_document.close()
    return images


def read_barcode_from_pdf(pdf_path):
    images = extract_images_from_pdf(pdf_path)

    for img in images:
        open_cv_image = np.array(img)

        if len(open_cv_image.shape) < 2:
            continue

        if len(open_cv_image.shape) == 2:
            gray = open_cv_image
        elif len(open_cv_image.shape) == 3:
            if open_cv_image.shape[2] == 4:
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        else:
            continue

        try:
            barcodes = decode(gray, symbols=[ZBarSymbol.CODE128])

            if not barcodes:
                denoised = cv2.fastNlMeansDenoising(gray)
                thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                barcodes = decode(thresh, symbols=[ZBarSymbol.CODE128])

            # バーコードが見つかった場合、左上のバーコードを選択
            if barcodes:
                top_left_barcode = min(barcodes, key=lambda b: b.rect.top + b.rect.left)
                barcode_data = top_left_barcode.data.decode('utf-8')
                return barcode_data

        except Exception as e:
            print(f"バーコード読み取り中にエラーが発生しました: {str(e)}")
            continue

    return None


def open_error_folder(error_dir):
    try:
        if os.name == 'nt':  # Windows
            os.startfile(error_dir)
        elif os.name == 'posix':  # macOS と Linux
            subprocess.call(['open', error_dir])
        else:
            print(f"エラーフォルダを開けません: {error_dir}")
    except Exception as e:
        print(f"エラーフォルダを開く際にエラーが発生しました: {str(e)}")


def process_pdf(pdf_path, error_dir, done_dir, status_callback, logger, config):
    try:
        if not os.path.exists(pdf_path):
            message = f"ファイルが見つかりません: {pdf_path}"
            logger.warning(message)
            status_callback(message)
            return

        logger.info(f"PDFの処理を開始: {pdf_path}")
        barcode_data = read_barcode_from_pdf(pdf_path)

        if barcode_data:
            new_filename = f"{barcode_data}.pdf"
            new_path = os.path.join(done_dir, new_filename)
            shutil.move(pdf_path, new_path)
            message = f"処理完了: {os.path.basename(pdf_path)} -> {new_filename}"
            logger.info(message)
            status_callback(message)
        else:
            message = f"{os.path.basename(pdf_path)} からバーコードが見つかりませんでした"
            logger.warning(message)
            status_callback(message)
            error_path = os.path.join(error_dir, os.path.basename(pdf_path))
            shutil.move(pdf_path, error_path)
            message = f"{os.path.basename(pdf_path)} をエラーフォルダーに移動しました"
            logger.info(message)
            status_callback(message)
            if config.auto_open_error_folder:
                open_error_folder(error_dir)

    except Exception as e:
        message = f"{os.path.basename(pdf_path)} の処理中にエラーが発生しました: {str(e)}"
        logger.error(message, exc_info=True)
        status_callback(message)
        try:
            if os.path.exists(pdf_path):
                error_path = os.path.join(error_dir, os.path.basename(pdf_path))
                shutil.move(pdf_path, error_path)
                message = f"{os.path.basename(pdf_path)} をエラーフォルダーに移動しました"
                logger.info(message)
                status_callback(message)
                if config.auto_open_error_folder:
                    open_error_folder(error_dir)
        except Exception as move_error:
            logger.error(f"ファイルの移動中にエラーが発生しました: {str(move_error)}", exc_info=True)
            status_callback(f"ファイルの移動中にエラーが発生しました: {str(move_error)}")


class PDFHandler(FileSystemEventHandler):
    def __init__(self, processing_dir, error_dir, done_dir, status_callback, logger, config):
        self.processing_dir = processing_dir
        self.error_dir = error_dir
        self.done_dir = done_dir
        self.status_callback = status_callback
        self.logger = logger
        self.config = config

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            self.logger.info(f"新しいPDFファイルを検出: {event.src_path}")
            self.status_callback(f"新しいPDFファイルを検出しました: {event.src_path}")
            time.sleep(1)
            try:
                process_pdf(event.src_path, self.error_dir, self.done_dir, self.status_callback, self.logger,
                            self.config)
            except Exception as e:
                self.logger.error(f"PDFの処理中にエラーが発生しました: {str(e)}", exc_info=True)
                self.status_callback(f"PDFの処理中にエラーが発生しました: {str(e)}")


class PDFProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("BarcodePDF")
        self.config = Config()
        self.master.geometry(f"{self.config.ui_width}x{self.config.ui_height}")

        self.logger = setup_logger(self.config)
        self.create_widgets()
        self.observer = None
        self.is_watching = False

        self.process_existing_pdfs()
        self.start_watching()

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        if self.config.start_minimized:
            self.master.iconify()

    def create_widgets(self):
        self.frame = ttk.Frame(self.master, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        version_info = f"version: {VERSION} (最終更新日: {LAST_UPDATED})"
        ttk.Label(self.frame, text=version_info, font=("", 10, "bold")).grid(column=0, row=0, columnspan=3, sticky=tk.W)

        ttk.Label(self.frame, text="処理フォルダ:").grid(column=0, row=1, sticky=tk.W)
        self.processing_dir_label = ttk.Label(self.frame, text=self.config.processing_dir, width=50)
        self.processing_dir_label.grid(column=1, row=1, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="参照", command=lambda: self.browse_directory('processing_dir')).grid(column=2, row=1)

        ttk.Label(self.frame, text="エラーフォルダ:").grid(column=0, row=2, sticky=tk.W)
        self.error_dir_label = ttk.Label(self.frame, text=self.config.error_dir, width=50)
        self.error_dir_label.grid(column=1, row=2, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="参照", command=lambda: self.browse_directory('error_dir')).grid(column=2, row=2)

        ttk.Label(self.frame, text="完了フォルダ:").grid(column=0, row=3, sticky=tk.W)
        self.done_dir_label = ttk.Label(self.frame, text=self.config.done_dir, width=50)
        self.done_dir_label.grid(column=1, row=3, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="参照", command=lambda: self.browse_directory('done_dir')).grid(column=2, row=3)

        ttk.Label(self.frame, text="ログフォルダ:").grid(column=0, row=4, sticky=tk.W)
        self.log_dir_label = ttk.Label(self.frame, text=self.config.log_dir, width=50)
        self.log_dir_label.grid(column=1, row=4, sticky=(tk.W, tk.E))
        ttk.Button(self.frame, text="参照", command=lambda: self.browse_directory('log_dir')).grid(column=2, row=4)

        self.auto_open_var = tk.BooleanVar(value=self.config.auto_open_error_folder)
        self.auto_open_checkbox = ttk.Checkbutton(
            self.frame,
            text="エラーフォルダを自動的に開く",
            variable=self.auto_open_var
        )
        self.auto_open_checkbox.grid(column=0, row=5, columnspan=2, sticky=tk.W)

        ttk.Button(self.frame, text="設定を保存", command=self.save_config).grid(column=2, row=5, sticky=tk.E)

        self.exit_button = ttk.Button(self.frame, text="閉じる", command=self.quit_app)
        self.exit_button.grid(column=2, row=6, sticky=tk.E)

        ttk.Label(self.frame, text="ステータス:").grid(column=0, row=7, sticky=tk.W)

        self.status_text = tk.Text(self.frame, height=10, width=70, wrap=tk.WORD)
        self.status_text.grid(column=0, row=8, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.status_text.config(state=tk.DISABLED)

        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(column=3, row=8, sticky=(tk.N, tk.S))
        self.status_text['yscrollcommand'] = scrollbar.set

        for child in self.frame.winfo_children():
            child.grid_configure(padx=5, pady=5)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(8, weight=1)

    def browse_directory(self, dir_type):
        directory = filedialog.askdirectory()
        if directory:
            if dir_type == 'processing_dir':
                self.processing_dir_label.config(text=directory)
            elif dir_type == 'error_dir':
                self.error_dir_label.config(text=directory)
            elif dir_type == 'done_dir':
                self.done_dir_label.config(text=directory)
            elif dir_type == 'log_dir':
                self.log_dir_label.config(text=directory)

    def save_config(self):
        self.config.processing_dir = self.processing_dir_label['text']
        self.config.error_dir = self.error_dir_label['text']
        self.config.done_dir = self.done_dir_label['text']
        self.config.log_dir = self.log_dir_label['text']
        self.config.config['Directories'] = {
            'processing_dir': self.config.processing_dir,
            'error_dir': self.config.error_dir,
            'done_dir': self.config.done_dir,
            'log_dir': self.config.log_dir
        }
        self.config.auto_open_error_folder = self.auto_open_var.get()
        self.config.config['Options'] = {
            'auto_open_error_folder': str(self.config.auto_open_error_folder)
        }
        self.config.save()
        self.logger = setup_logger(self.config)
        messagebox.showinfo("設定保存", "設定が保存されました。")
        self.logger.info("設定が更新されました")

    @staticmethod
    def open_error_folder(error_dir):
        try:
            if os.name == 'nt':  # Windows
                os.startfile(error_dir)
            elif os.name == 'posix':  # macOS と Linux
                subprocess.call(['open', error_dir])
            else:
                print(f"エラーフォルダを開けません: {error_dir}")
        except Exception as e:
            print(f"エラーフォルダを開く際にエラーが発生しました: {str(e)}")

    def process_existing_pdfs(self):
        self.update_status("フォルダ内のPDFファイルを処理しています...")
        self.logger.info("既存のPDFファイルの処理を開始")
        for filename in os.listdir(self.config.processing_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.config.processing_dir, filename)
                process_pdf(pdf_path, self.config.error_dir, self.config.done_dir, self.update_status, self.logger,
                            self.config)

        self.update_status("フォルダ内のPDFファイルの処理が完了しました。")
        self.logger.info("既存のPDFファイルの処理が完了")

    def start_watching(self):
        if not self.is_watching:
            self.observer = Observer()
            event_handler = PDFHandler(
                self.config.processing_dir,
                self.config.error_dir,
                self.config.done_dir,
                self.update_status,
                self.logger,
                self.config,
            )
            self.observer.schedule(event_handler, self.config.processing_dir, recursive=False)
            self.observer.start()
            self.is_watching = True
            message = f"{self.config.processing_dir} の監視を開始しました..."
            self.update_status(message)
            self.logger.info(message)

    def stop_watching(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.is_watching = False
            message = "監視を停止しました。"
            self.update_status(message)
            self.logger.info(message)

    def update_status(self, message):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        self.logger.info(message)

    def quit_app(self):
        self.stop_watching()
        self.logger.info("アプリケーションを終了します")
        self.master.quit()

    def on_closing(self):
        if messagebox.askokcancel("終了", "アプリケーションを終了しますか？"):
            self.quit_app()


if __name__ == "__main__":
    root = tk.Tk()
    app = PDFProcessorApp(root)
    root.mainloop()
