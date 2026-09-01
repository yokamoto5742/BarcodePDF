"""PDFからCODE128バーコードを読み取る"""

import io
import logging

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageEnhance
from pyzbar.pyzbar import decode
from pyzbar.wrapper import ZBarSymbol

from utils.constants import MSG_BARCODE_READ_ERROR

logger = logging.getLogger(__name__)

CONTRAST_FACTOR = 2.0


def _to_enhanced_grayscale(image: Image.Image) -> Image.Image:
    gray_image = image.convert('L')
    return ImageEnhance.Contrast(gray_image).enhance(CONTRAST_FACTOR)


def extract_images_from_pdf(pdf_path: str) -> list[Image.Image]:
    images: list[Image.Image] = []

    # 破損PDFで例外が起きてもファイルハンドルを解放するためwith文を使う
    with pymupdf.open(pdf_path) as pdf_document:
        for page in pdf_document:
            for img in page.get_images(full=True):
                base_image = pdf_document.extract_image(img[0])
                embedded_image = Image.open(io.BytesIO(base_image["image"]))
                images.append(_to_enhanced_grayscale(embedded_image))

            # 埋め込み画像から読めない場合に備え、ページ全体のレンダリング結果も対象にする
            pix = page.get_pixmap()
            page_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(_to_enhanced_grayscale(page_image))

    return images


def _decode_code128(gray: np.ndarray) -> str | None:
    barcodes = decode(gray, symbols=[ZBarSymbol.CODE128])

    if not barcodes:
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        barcodes = decode(thresh, symbols=[ZBarSymbol.CODE128])

    if not barcodes:
        return None

    # 複数見つかった場合は最も左上にあるものを採用する
    top_left_barcode = min(barcodes, key=lambda b: b.rect.top + b.rect.left)
    return top_left_barcode.data.decode('utf-8')


def read_barcode_from_pdf(pdf_path: str) -> str | None:
    for image in extract_images_from_pdf(pdf_path):
        try:
            # extract_images_from_pdf がグレースケール化済みのため2次元配列になる
            barcode_data = _decode_code128(np.array(image))
            if barcode_data:
                return barcode_data
        except Exception as e:
            logger.warning(MSG_BARCODE_READ_ERROR.format(error=str(e)))
            continue

    return None
