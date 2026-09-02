"""P0/P1: PDF1ページ目上部からのCODE128読み取り

pyzbarのネイティブデコードはモックする（実PDFでの読み取り確認はWindows実機で行う）。
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest
from pytest_mock import MockerFixture

from service.barcode_reader import (
    MIN_BARCODE_WIDTH_RATIO,
    RENDER_ZOOM,
    TOP_BAND_RATIO,
    _decode_code128,
    _render_top_band,
    _select_widest_barcode,
    read_barcode_from_pdf,
)

PAGE_SIZE = 200


def make_pdf(path: Path, pages: int = 1) -> str:
    """上部の帯の左半分だけを灰色（153）で塗ったPDFを生成する"""
    document = pymupdf.open()

    for _ in range(pages):
        page = document.new_page(width=PAGE_SIZE, height=PAGE_SIZE)
        band_height = PAGE_SIZE * TOP_BAND_RATIO
        page.draw_rect(pymupdf.Rect(0, 0, PAGE_SIZE / 2, band_height), fill=(0.6, 0.6, 0.6))

    document.save(str(path))
    document.close()
    return str(path)


def fake_barcode(data: bytes, width: int) -> SimpleNamespace:
    return SimpleNamespace(data=data, rect=SimpleNamespace(width=width))


# --- _render_top_band（P1） ---


def test_render_top_band_crops_to_top_ratio(tmp_path: Path) -> None:
    gray = _render_top_band(make_pdf(tmp_path / 'one.pdf'))

    assert gray is not None
    # 高さのみTOP_BAND_RATIOで切り取り、幅はページ全体をRENDER_ZOOM倍で描画する
    assert gray.shape == (
        int(PAGE_SIZE * TOP_BAND_RATIO * RENDER_ZOOM),
        int(PAGE_SIZE * RENDER_ZOOM),
    )


def test_render_top_band_returns_grayscale_array(tmp_path: Path) -> None:
    gray = _render_top_band(make_pdf(tmp_path / 'one.pdf'))

    assert gray is not None
    assert gray.ndim == 2
    assert gray.dtype == np.uint8


def test_render_top_band_enhances_contrast(tmp_path: Path) -> None:
    gray = _render_top_band(make_pdf(tmp_path / 'one.pdf'))

    assert gray is not None
    middle_row = gray[gray.shape[0] // 2]
    # 平均より暗い灰色(153)はさらに暗く、白(255)は上限で頭打ちになる
    assert middle_row[10] < 153
    assert middle_row[-10] == 255


def test_render_top_band_reads_only_first_page(tmp_path: Path, mocker: MockerFixture) -> None:
    pages = pymupdf.open(make_pdf(tmp_path / 'multi.pdf', pages=3))
    document = mocker.MagicMock()
    document.__enter__.return_value = document
    document.page_count = 3
    document.__getitem__.side_effect = lambda index: pages[index]
    mocker.patch('service.barcode_reader.pymupdf.open', return_value=document)

    _render_top_band('any.pdf')

    document.__getitem__.assert_called_once_with(0)


def test_render_top_band_returns_none_for_empty_pdf(mocker: MockerFixture) -> None:
    document = mocker.MagicMock()
    document.__enter__.return_value = document
    document.page_count = 0
    mocker.patch('service.barcode_reader.pymupdf.open', return_value=document)

    assert _render_top_band('empty.pdf') is None
    document.__getitem__.assert_not_called()


def test_render_top_band_releases_handle_on_error(mocker: MockerFixture) -> None:
    """破損PDFで例外が起きてもファイルハンドルを解放する"""
    document = mocker.MagicMock()
    document.__enter__.return_value = document
    type(document).page_count = mocker.PropertyMock(side_effect=RuntimeError('破損PDF'))
    mocker.patch('service.barcode_reader.pymupdf.open', return_value=document)

    with pytest.raises(RuntimeError):
        _render_top_band('broken.pdf')

    document.__exit__.assert_called_once()


def test_render_top_band_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        _render_top_band(str(tmp_path / 'missing.pdf'))


# --- _decode_code128（P0） ---


def test_decode_returns_barcodes_found_on_first_attempt(mocker: MockerFixture) -> None:
    barcode = fake_barcode(b'ABC123', 100)
    decode = mocker.patch('service.barcode_reader.decode', return_value=[barcode])
    denoise = mocker.patch('service.barcode_reader.cv2.fastNlMeansDenoising')

    assert _decode_code128(np.zeros((4, 4), dtype=np.uint8)) == [barcode]
    assert decode.call_count == 1
    denoise.assert_not_called()


def test_decode_retries_with_binarized_image(mocker: MockerFixture) -> None:
    gray = np.zeros((4, 4), dtype=np.uint8)
    binarized = np.ones((4, 4), dtype=np.uint8)
    barcode = fake_barcode(b'RETRY', 100)
    decode = mocker.patch('service.barcode_reader.decode', side_effect=[[], [barcode]])
    denoise = mocker.patch('service.barcode_reader.cv2.fastNlMeansDenoising', return_value=gray)
    threshold = mocker.patch('service.barcode_reader.cv2.threshold', return_value=(0, binarized))

    assert _decode_code128(gray) == [barcode]
    denoise.assert_called_once()
    threshold.assert_called_once()
    assert decode.call_args_list[1].args[0] is binarized


def test_decode_returns_empty_when_both_attempts_fail(mocker: MockerFixture) -> None:
    mocker.patch('service.barcode_reader.decode', return_value=[])
    mocker.patch('service.barcode_reader.cv2.fastNlMeansDenoising', return_value=np.zeros((4, 4)))
    mocker.patch('service.barcode_reader.cv2.threshold', return_value=(0, np.zeros((4, 4))))

    assert _decode_code128(np.zeros((4, 4), dtype=np.uint8)) == []


def test_decode_limits_symbols_to_code128(mocker: MockerFixture) -> None:
    decode = mocker.patch('service.barcode_reader.decode', return_value=[])
    mocker.patch('service.barcode_reader.cv2.fastNlMeansDenoising', return_value=np.zeros((4, 4)))
    mocker.patch('service.barcode_reader.cv2.threshold', return_value=(0, np.zeros((4, 4))))

    _decode_code128(np.zeros((4, 4), dtype=np.uint8))

    # QRなど他の種類は対象にしない
    assert [symbol.name for symbol in decode.call_args.kwargs['symbols']] == ['CODE128']


# --- _select_widest_barcode（P0） ---


def test_select_returns_widest_barcode() -> None:
    barcodes = [
        fake_barcode(b'NARROW', 300),
        fake_barcode(b'WIDEST', 500),
        fake_barcode(b'MIDDLE', 400),
    ]

    assert _select_widest_barcode(barcodes, 1000) == 'WIDEST'


def test_select_ignores_barcodes_below_minimum_width() -> None:
    """帯の中に紛れた小さなバーコードは幅で除外する"""
    minimum = 1000 * MIN_BARCODE_WIDTH_RATIO
    barcodes = [fake_barcode(b'SMALL', int(minimum) - 1), fake_barcode(b'LARGE', int(minimum))]

    assert _select_widest_barcode(barcodes, 1000) == 'LARGE'


def test_select_returns_none_when_all_barcodes_are_small() -> None:
    barcodes = [fake_barcode(b'SMALL', 10), fake_barcode(b'TINY', 5)]

    assert _select_widest_barcode(barcodes, 1000) is None


def test_select_returns_none_for_empty_list() -> None:
    assert _select_widest_barcode([], 1000) is None


def test_select_keeps_first_barcode_when_widths_tie() -> None:
    barcodes = [fake_barcode(b'FIRST', 500), fake_barcode(b'SECOND', 500)]

    assert _select_widest_barcode(barcodes, 1000) == 'FIRST'


def test_select_raises_for_non_utf8_payload() -> None:
    with pytest.raises(UnicodeDecodeError):
        _select_widest_barcode([fake_barcode(b'\xff\xfe', 500)], 1000)


# --- read_barcode_from_pdf（P0） ---


@pytest.fixture
def rendered_band(mocker: MockerFixture) -> None:
    mocker.patch(
        'service.barcode_reader._render_top_band',
        return_value=np.zeros((100, 1000), dtype=np.uint8),
    )


def test_read_barcode_returns_widest_barcode_in_top_band(
    rendered_band: None,
    mocker: MockerFixture,
) -> None:
    mocker.patch('service.barcode_reader._decode_code128', return_value=[
        fake_barcode(b'SMALL', 50),
        fake_barcode(b'LARGE', 500),
    ])

    assert read_barcode_from_pdf('any.pdf') == 'LARGE'


def test_read_barcode_returns_none_when_nothing_found(
    rendered_band: None,
    mocker: MockerFixture,
) -> None:
    mocker.patch('service.barcode_reader._decode_code128', return_value=[])

    assert read_barcode_from_pdf('any.pdf') is None


def test_read_barcode_returns_none_for_pdf_without_pages(mocker: MockerFixture) -> None:
    mocker.patch('service.barcode_reader._render_top_band', return_value=None)
    decode = mocker.patch('service.barcode_reader._decode_code128')

    assert read_barcode_from_pdf('any.pdf') is None
    decode.assert_not_called()


def test_read_barcode_logs_and_returns_none_on_decode_error(
    rendered_band: None,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mocker.patch(
        'service.barcode_reader._decode_code128',
        side_effect=UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid'),
    )

    assert read_barcode_from_pdf('any.pdf') is None
    assert caplog.records[0].levelno == logging.WARNING


def test_read_barcode_propagates_render_error(mocker: MockerFixture) -> None:
    """PDF展開の失敗はprocess_pdf側でエラーフォルダ行きとして扱う"""
    mocker.patch('service.barcode_reader._render_top_band', side_effect=RuntimeError('破損PDF'))

    with pytest.raises(RuntimeError):
        read_barcode_from_pdf('any.pdf')
