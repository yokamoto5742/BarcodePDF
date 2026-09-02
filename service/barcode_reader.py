"""PDF1ページ目の上部にある大きなCODE128バーコードを読み取る"""

import logging

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageEnhance
from pyzbar.pyzbar import Decoded, decode
from pyzbar.wrapper import ZBarSymbol

from utils.constants import MSG_BARCODE_READ_ERROR

logger = logging.getLogger(__name__)

CONTRAST_FACTOR = 2.0

# 72dpi基準の拡大率。等倍ではバーの太さが足りずデコードできない
RENDER_ZOOM = 3.0

# ページ上端から探索する高さの割合。これより下の小さなバーコードやQRは画像に含めない
TOP_BAND_RATIO = 0.15

# ページ幅に対する最小幅。帯の中に小さなバーコードが並んでいても大きい方だけを採用する
MIN_BARCODE_WIDTH_RATIO = 0.20


def _render_top_band(pdf_path: str) -> np.ndarray | None:
    """1ページ目の上部だけを高解像度でレンダリングしてグレースケール配列にする"""
    # 破損PDFで例外が起きてもファイルハンドルを解放するためwith文を使う
    with pymupdf.open(pdf_path) as pdf_document:
        if pdf_document.page_count == 0:
            return None

        page = pdf_document[0]
        rect = page.rect
        clip = pymupdf.Rect(rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * TOP_BAND_RATIO)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM), clip=clip)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    gray_image = ImageEnhance.Contrast(image.convert('L')).enhance(CONTRAST_FACTOR)
    return np.array(gray_image)


def _decode_code128(gray: np.ndarray) -> list[Decoded]:
    barcodes = decode(gray, symbols=[ZBarSymbol.CODE128])

    if not barcodes:
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        barcodes = decode(thresh, symbols=[ZBarSymbol.CODE128])

    return barcodes


def _select_widest_barcode(barcodes: list[Decoded], image_width: int) -> str | None:
    """幅がページ幅の一定割合以上のもののうち最も広いものを採用する"""
    min_width = image_width * MIN_BARCODE_WIDTH_RATIO
    wide_barcodes = [barcode for barcode in barcodes if barcode.rect.width >= min_width]

    if not wide_barcodes:
        return None

    return max(wide_barcodes, key=lambda barcode: barcode.rect.width).data.decode('utf-8')


def read_barcode_from_pdf(pdf_path: str) -> str | None:
    gray = _render_top_band(pdf_path)
    if gray is None:
        return None

    try:
        return _select_widest_barcode(_decode_code128(gray), gray.shape[1])
    except Exception as e:
        logger.warning(MSG_BARCODE_READ_ERROR.format(error=str(e)))
        return None
