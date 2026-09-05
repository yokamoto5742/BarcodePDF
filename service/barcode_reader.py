"""PDF1ページ目の上部にある大きなCODE128バーコードを読み取る"""

import logging

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageEnhance
from pyzbar.pyzbar import Decoded, decode
from pyzbar.wrapper import ZBarSymbol

from utils.config_manager import AppConfig
from utils.constants import MSG_BARCODE_READ_ERROR

logger = logging.getLogger(__name__)


def _render_top_band(pdf_path: str, config: AppConfig) -> np.ndarray | None:
    """1ページ目の上部だけを高解像度でレンダリングしてグレースケール配列にする"""
    with pymupdf.open(pdf_path) as pdf_document:
        if pdf_document.page_count == 0:
            return None

        page = pdf_document[0]
        rect = page.rect
        clip = pymupdf.Rect(
            rect.x0, rect.y0, rect.x1, rect.y0 + rect.height * config.top_band_ratio
        )
        matrix = pymupdf.Matrix(config.render_zoom, config.render_zoom)
        pix = page.get_pixmap(matrix=matrix, clip=clip)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    gray_image = ImageEnhance.Contrast(image.convert('L')).enhance(config.contrast_factor)
    return np.array(gray_image)


def _decode_code128(gray: np.ndarray) -> list[Decoded]:
    barcodes = decode(gray, symbols=[ZBarSymbol.CODE128])

    if not barcodes:
        denoised = cv2.fastNlMeansDenoising(gray)
        thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        barcodes = decode(thresh, symbols=[ZBarSymbol.CODE128])

    return barcodes


def _select_widest_barcode(
    barcodes: list[Decoded],
    image_width: int,
    min_width_ratio: float,
) -> str | None:
    """幅がページ幅の一定割合以上のもののうち最も広いものを採用する"""
    min_width = image_width * min_width_ratio
    wide_barcodes = [barcode for barcode in barcodes if barcode.rect.width >= min_width]

    if not wide_barcodes:
        return None

    return max(wide_barcodes, key=lambda barcode: barcode.rect.width).data.decode('utf-8')


def read_barcode_from_pdf(pdf_path: str, config: AppConfig) -> str | None:
    gray = _render_top_band(pdf_path, config)
    if gray is None:
        return None

    try:
        return _select_widest_barcode(
            _decode_code128(gray), gray.shape[1], config.min_barcode_width_ratio
        )
    except Exception as e:
        logger.warning(MSG_BARCODE_READ_ERROR.format(error=str(e)))
        return None
