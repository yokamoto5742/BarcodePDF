import os
import io
import cv2
import numpy as np
from PIL import Image
from pyzbar.pyzbar import decode
import fitz
import shutil
import configparser
import tkinter as tk
from tkinter import filedialog, messagebox
import threading


class PDFBarcodeApp:
    def __init__(self, master):
        self.master = master
        master.title("PDFバーコード処理アプリ")
        master.geometry("400x300")

        self.processing_dir, self.error_dir, self.done_dir = self.load_config()

        self.label = tk.Label(master, text="処理するPDFファイルを選択してください")
        self.label.pack(pady=10)

        self.select_button = tk.Button(master, text="フォルダ選択", command=self.select_folder)
        self.select_button.pack(pady=5)

        self.process_button = tk.Button(master, text="処理開始", command=self.start_processing, state=tk.NORMAL)
        self.process_button.pack(pady=5)

        self.progress_label = tk.Label(master, text="")
        self.progress_label.pack(pady=10)

        self.status_label = tk.Label(master, text="")
        self.status_label.pack(pady=5)

        # 初期フォルダを設定
        self.label.config(text=f"選択されたフォルダ: {self.processing_dir}")

    def load_config(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        if 'Directories' not in config:
            raise ValueError("config.iniファイルにDirectoriesセクションがありません。")

        processing_dir = config['Directories'].get('processing_dir')
        error_dir = config['Directories'].get('error_dir')
        done_dir = config['Directories'].get('done_dir')

        if not all([processing_dir, error_dir, done_dir]):
            raise ValueError("config.iniファイルに必要な設定が不足しています。")

        return processing_dir, error_dir, done_dir

    def select_folder(self):
        selected_dir = filedialog.askdirectory(initialdir=self.processing_dir)
        if selected_dir:
            self.processing_dir = selected_dir
            self.label.config(text=f"選択されたフォルダ: {self.processing_dir}")

    def start_processing(self):
        self.process_button.config(state=tk.DISABLED)
        self.progress_label.config(text="処理中...")
        threading.Thread(target=self.rename_pdf_with_barcode, daemon=True).start()

    def extract_images_from_pdf(self, pdf_path):
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

    def read_barcode_from_pdf(self, pdf_path):
        images = self.extract_images_from_pdf(pdf_path)

        for img in images:
            open_cv_image = np.array(img)
            if open_cv_image.shape[2] == 4:  # RGBAの場合
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGBA2BGR)
            else:
                open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

            barcodes = decode(gray)

            for barcode in barcodes:
                barcode_data = barcode.data.decode('utf-8')
                return barcode_data

        return None

    def rename_pdf_with_barcode(self):
        os.makedirs(self.error_dir, exist_ok=True)
        os.makedirs(self.done_dir, exist_ok=True)

        for filename in os.listdir(self.processing_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.processing_dir, filename)
                try:
                    barcode_data = self.read_barcode_from_pdf(pdf_path)

                    if barcode_data:
                        new_filename = f"{barcode_data}.pdf"
                        new_path = os.path.join(self.done_dir, new_filename)
                        shutil.move(pdf_path, new_path)
                        self.update_status(f"処理完了: {filename} -> {new_filename}")
                    else:
                        self.update_status(f"{filename} からバーコードが見つかりませんでした")
                        error_path = os.path.join(self.error_dir, filename)
                        shutil.move(pdf_path, error_path)
                        self.update_status(f"{filename} をエラーフォルダーに移動しました")
                except Exception as e:
                    self.update_status(f"{filename} の処理中にエラーが発生しました: {str(e)}")
                    error_path = os.path.join(self.error_dir, filename)
                    shutil.move(pdf_path, error_path)
                    self.update_status(f"{filename} をエラーフォルダーに移動しました")

        self.update_status("すべてのファイルの処理が完了しました")
        self.process_button.config(state=tk.NORMAL)

    def update_status(self, message):
        self.master.after(0, lambda: self.status_label.config(text=message))


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = PDFBarcodeApp(root)
        root.mainloop()
    except ValueError as e:
        messagebox.showerror("設定エラー", str(e))
        root.destroy()
