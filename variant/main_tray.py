import os
import io
import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode
import fitz
import shutil
import configparser
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pystray
import threading


class Config:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.processing_dir = self.config['Directories']['processing_dir']
        self.error_dir = self.config['Directories']['error_dir']
        self.done_dir = self.config['Directories']['done_dir']

    def save(self):
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)


def extract_images_from_pdf(pdf_path):
    images = []
    pdf_document = fitz.open(pdf_path)

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)

    pdf_document.close()
    return images


def read_barcode_from_pdf(pdf_path):
    images = extract_images_from_pdf(pdf_path)

    for img in images:
        open_cv_image = np.array(img)
        if open_cv_image.shape[2] == 4:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
        else:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        barcodes = decode(gray)

        for barcode in barcodes:
            barcode_data = barcode.data.decode('utf-8')
            return barcode_data

    return None


def process_pdf(pdf_path, error_dir, done_dir, status_callback):
    try:
        barcode_data = read_barcode_from_pdf(pdf_path)

        if barcode_data:
            new_filename = f"{barcode_data}.pdf"
            new_path = os.path.join(done_dir, new_filename)
            shutil.move(pdf_path, new_path)
            status_callback(f"処理完了: {os.path.basename(pdf_path)} -> {new_filename}")
        else:
            status_callback(f"{os.path.basename(pdf_path)} からバーコードが見つかりませんでした")
            error_path = os.path.join(error_dir, os.path.basename(pdf_path))
            shutil.move(pdf_path, error_path)
            status_callback(f"{os.path.basename(pdf_path)} をエラーフォルダーに移動しました")
    except Exception as e:
        status_callback(f"{os.path.basename(pdf_path)} の処理中にエラーが発生しました: {str(e)}")
        error_path = os.path.join(error_dir, os.path.basename(pdf_path))
        shutil.move(pdf_path, error_path)
        status_callback(f"{os.path.basename(pdf_path)} をエラーフォルダーに移動しました")


class PDFHandler(FileSystemEventHandler):
    def __init__(self, processing_dir, error_dir, done_dir, status_callback):
        self.processing_dir = processing_dir
        self.error_dir = error_dir
        self.done_dir = done_dir
        self.status_callback = status_callback

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            self.status_callback(f"新しいPDFファイルを検出しました: {event.src_path}")
            time.sleep(1)
            process_pdf(event.src_path, self.error_dir, self.done_dir, self.status_callback)


class PDFProcessorApp:
    def __init__(self):
        self.config = Config()
        self.observer = None
        self.is_watching = False
        self.root = None
        self.icon = None
        self.start_watching()

    def create_window(self):
        self.root = tk.Tk()
        self.root.title("RenamePDF")
        self.root.geometry("600x400")
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.hide_window)

    def create_widgets(self):
        self.frame = ttk.Frame(self.root, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(self.frame, text="処理フォルダ:").grid(column=0, row=0, sticky=tk.W)
        self.processing_dir_entry = ttk.Entry(self.frame, width=50)
        self.processing_dir_entry.grid(column=1, row=0, sticky=(tk.W, tk.E))
        self.processing_dir_entry.insert(0, self.config.processing_dir)
        ttk.Button(self.frame, text="参照", command=lambda: self.browse_directory('processing_dir')).grid(column=2, row=0)

        ttk.Label(self.frame, text="エラーフォルダ:").grid(column=0, row=1, sticky=tk.W)
        self.error_dir_entry = ttk.Entry(self.frame, width=50)
        self.error_dir_entry.grid(column=1, row=1, sticky=(tk.W, tk.E))
        self.error_dir_entry.insert(0, self.config.error_dir)
        ttk.Button(self.frame, text="参照", command=lambda: self.browse_directory('error_dir')).grid(column=2, row=1)

        ttk.Label(self.frame, text="完了フォルダ:").grid(column=0, row=2, sticky=tk.W)
        self.done_dir_entry = ttk.Entry(self.frame, width=50)
        self.done_dir_entry.grid(column=1, row=2, sticky=(tk.W, tk.E))
        self.done_dir_entry.insert(0, self.config.done_dir)
        ttk.Button(self.frame, text="参照", command=lambda: self.browse_directory('done_dir')).grid(column=2, row=2)

        ttk.Button(self.frame, text="設定を保存", command=self.save_config).grid(column=1, row=3, sticky=tk.E)

        self.watch_button = ttk.Button(self.frame, text="監視停止", command=self.toggle_watch)
        self.watch_button.grid(column=1, row=4, sticky=tk.E)

        self.status_text = tk.Text(self.frame, height=10, width=70, wrap=tk.WORD)
        self.status_text.grid(column=0, row=5, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.status_text.config(state=tk.DISABLED)

        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.status_text.yview)
        scrollbar.grid(column=3, row=5, sticky=(tk.N, tk.S))
        self.status_text['yscrollcommand'] = scrollbar.set

        for child in self.frame.winfo_children():
            child.grid_configure(padx=5, pady=5)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(5, weight=1)

    def browse_directory(self, dir_type):
        directory = filedialog.askdirectory()
        if directory:
            if dir_type == 'processing_dir':
                self.processing_dir_entry.delete(0, tk.END)
                self.processing_dir_entry.insert(0, directory)
            elif dir_type == 'error_dir':
                self.error_dir_entry.delete(0, tk.END)
                self.error_dir_entry.insert(0, directory)
            elif dir_type == 'done_dir':
                self.done_dir_entry.delete(0, tk.END)
                self.done_dir_entry.insert(0, directory)

    def save_config(self):
        self.config.processing_dir = self.processing_dir_entry.get()
        self.config.error_dir = self.error_dir_entry.get()
        self.config.done_dir = self.done_dir_entry.get()
        self.config.config['Directories'] = {
            'processing_dir': self.config.processing_dir,
            'error_dir': self.config.error_dir,
            'done_dir': self.config.done_dir
        }
        self.config.save()
        messagebox.showinfo("設定保存", "設定が保存されました。")

    def toggle_watch(self):
        if self.is_watching:
            self.stop_watching()
        else:
            self.start_watching()

    def start_watching(self):
        if not self.is_watching:
            self.observer = Observer()
            event_handler = PDFHandler(
                self.config.processing_dir,
                self.config.error_dir,
                self.config.done_dir,
                self.update_status
            )
            self.observer.schedule(event_handler, self.config.processing_dir, recursive=False)
            self.observer.start()
            self.is_watching = True
            if self.root:
                self.watch_button.config(text="監視停止")
            self.update_status(f"{self.config.processing_dir} の監視を開始しました...")

    def stop_watching(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.is_watching = False
            if self.root:
                self.watch_button.config(text="監視開始")
            self.update_status("監視を停止しました。")

    def update_status(self, message):
        print(message)  # コンソールに出力
        if self.root:
            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, message + "\n")
            self.status_text.see(tk.END)
            self.status_text.config(state=tk.DISABLED)

    def show_window(self):
        if not self.root:
            self.create_window()
        self.root.deiconify()

    def hide_window(self):
        self.root.withdraw()

    def quit_app(self):
        self.stop_watching()
        if self.root:
            self.root.quit()
        self.icon.stop()

    def create_menu(self):
        menu = (
            pystray.MenuItem('設定を開く', self.show_window),
            pystray.MenuItem('終了', self.quit_app)
        )
        return menu

    def run(self):
        image = Image.open("trayicon.png")  # アイコン画像のパスを指定
        self.icon = pystray.Icon("RenamePDF", image, "RenamePDF", self.create_menu())
        self.icon.run()


if __name__ == "__main__":
    app = PDFProcessorApp()
    app_thread = threading.Thread(target=app.run)
    app_thread.start()
    app_thread.join()
