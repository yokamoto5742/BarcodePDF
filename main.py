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
import tkinter as tk
from tkinter import ttk
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import subprocess
import threading

VERSION = "1.0.6"
LAST_UPDATED = "2024/08/08"

processed_files = {}

class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.processing_dir = self.config['Directories']['processing_dir']
        self.error_dir = self.config['Directories']['error_dir']
        self.done_dir = self.config['Directories']['done_dir']
        self.log_dir = self.config['Directories'].get('log_dir', 'log')
        self.ui_width = self.config['UI'].getint('width', 600)
        self.ui_height = self.config['UI'].getint('height', 400)
        self.auto_open_error_folder = self.config['Options'].getboolean('auto_open_error_folder', True)
        self.log_retention_days = self.config['Logging'].getint('retention_days', 14)

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
    global processed_files

    # ファイルの最終更新時刻を取得
    file_mtime = os.path.getmtime(pdf_path)

    # このファイルが既に処理済みで、最終更新時刻が同じ場合はスキップ
    if pdf_path in processed_files and processed_files[pdf_path] == file_mtime:
        logger.info(f"ファイルは既に処理済みです: {pdf_path}")
        return

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

        # 処理済みファイルを記録
        processed_files[pdf_path] = file_mtime

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


class PDFProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("BarcodePDF")
        self.config = Config()
        self.master.geometry(f"{self.config.ui_width}x{self.config.ui_height}")

        self.logger = setup_logger(self.config)
        self.create_widgets()
        self.is_watching = False
        self.watch_thread = None

        self.process_existing_pdfs()
        self.start_watching()

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.frame = ttk.Frame(self.master, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        version_info = f"version: {VERSION} (最終更新日: {LAST_UPDATED})"
        ttk.Label(self.frame, text=version_info, font=("", 10, "bold")).grid(column=0, row=0, columnspan=2, sticky=tk.W)

        ttk.Label(self.frame, text="ステータス:").grid(column=0, row=1, sticky=tk.W)

        self.status_text = tk.Text(self.frame, height=10, width=70, wrap=tk.WORD)
        self.status_text.grid(column=0, row=2, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.status_text.config(state=tk.DISABLED)

        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(column=2, row=2, sticky=(tk.N, tk.S))
        self.status_text['yscrollcommand'] = scrollbar.set

        self.exit_button = ttk.Button(self.frame, text="閉じる", command=self.quit_app)
        self.exit_button.grid(column=1, row=3, sticky=tk.E)

        for child in self.frame.winfo_children():
            child.grid_configure(padx=5, pady=5)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(2, weight=1)

    def process_existing_pdfs(self):
        self.update_status("フォルダ内のPDFファイルを処理しています...")
        self.logger.info("既存のPDFファイルの処理を開始")
        for filename in os.listdir(self.config.processing_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.config.processing_dir, filename)
                process_pdf(pdf_path, self.config.error_dir, self.config.done_dir, self.update_status, self.logger, self.config)

        self.update_status("フォルダ内のPDFファイルの処理が完了しました。")
        self.logger.info("既存のPDFファイルの処理が完了")

    def start_watching(self):
        if not self.is_watching:
            self.is_watching = True
            self.watch_thread = threading.Thread(target=self.watch_directory)
            self.watch_thread.start()
            message = f"{self.config.processing_dir} の監視を開始しました..."
            self.update_status(message)
            self.logger.info(message)

    def stop_watching(self):
        if self.is_watching:
            self.is_watching = False
            if self.watch_thread:
                self.watch_thread.join()
            message = "監視を停止しました。"
            self.update_status(message)
            self.logger.info(message)

    def watch_directory(self):
        while self.is_watching:
            for filename in os.listdir(self.config.processing_dir):
                if filename.lower().endswith('.pdf'):
                    pdf_path = os.path.join(self.config.processing_dir, filename)
                    process_pdf(pdf_path, self.config.error_dir, self.config.done_dir, self.update_status, self.logger, self.config)
            time.sleep(3)  # 3秒間待機

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
        self.quit_app()

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFProcessorApp(root)
    root.mainloop()
