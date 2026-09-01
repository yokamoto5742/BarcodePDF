"""P0/P1: PDFからのCODE128読み取り

pyzbarのネイティブデコードはモックする（実PDFでの読み取り確認はWindows実機で行う）。
"""

import io
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pymupdf
import pytest
from PIL import Image
from pytest_mock import MockerFixture

from service.barcode_reader import (
    CONTRAST_FACTOR,
    _decode_code128,
    _to_enhanced_grayscale,
    extract_images_from_pdf,
    read_barcode_from_pdf,
)


def make_pdf(path: Path, pages: int = 1, images_per_page: int = 1) -> str:
    """埋め込み画像を持つPDFを生成する"""
    document = pymupdf.open()
    buffer = io.BytesIO()
    Image.new('RGB', (50, 50), 'white').save(buffer, format='PNG')

    for _ in range(pages):
        page = document.new_page(width=200, height=200)
        for index in range(images_per_page):
            rect = pymupdf.Rect(10, 10 + index * 60, 60, 60 + index * 60)
            page.insert_image(rect, stream=buffer.getvalue())

    document.save(str(path))
    document.close()
    return str(path)


def fake_barcode(data: bytes, top: int, left: int) -> SimpleNamespace:
    return SimpleNamespace(data=data, rect=SimpleNamespace(top=top, left=left))


# --- _to_enhanced_grayscale（P2） ---


def test_to_enhanced_grayscale_returns_grayscale() -> None:
    result = _to_enhanced_grayscale(Image.new('RGB', (4, 4), (10, 200, 30)))

    assert result.mode == 'L'


def test_to_enhanced_grayscale_applies_contrast_factor() -> None:
    source = Image.new('L', (2, 1))
    source.putpixel((0, 0), 100)
    source.putpixel((1, 0), 156)

    result = _to_enhanced_grayscale(source)

    # 平均128を基準に (値 - 平均) * CONTRAST_FACTOR + 平均 へ変換される
    assert CONTRAST_FACTOR == 2.0
    assert np.array(result).flatten().tolist() == [72, 184]


# --- extract_images_from_pdf（P1） ---


def test_extract_images_returns_embedded_images_and_page_render(tmp_path: Path) -> None:
    images = extract_images_from_pdf(make_pdf(tmp_path / 'one.pdf'))

    # 埋め込み画像1枚 + ページ全体のレンダリング1枚
    assert len(images) == 2
    assert images[0].size == (50, 50)
    assert images[1].size == (200, 200)


def test_extract_images_converts_all_to_grayscale(tmp_path: Path) -> None:
    images = extract_images_from_pdf(make_pdf(tmp_path / 'one.pdf'))

    assert [image.mode for image in images] == ['L', 'L']


def test_extract_images_accumulates_over_pages(tmp_path: Path) -> None:
    images = extract_images_from_pdf(make_pdf(tmp_path / 'multi.pdf', pages=3))

    assert len(images) == 6


def test_extract_images_returns_only_render_when_no_embedded_image(tmp_path: Path) -> None:
    images = extract_images_from_pdf(make_pdf(tmp_path / 'blank.pdf', images_per_page=0))

    assert len(images) == 1


def test_extract_images_releases_handle_on_error(mocker: MockerFixture) -> None:
    """破損PDFで例外が起きてもファイルハンドルを解放する"""
    document = mocker.MagicMock()
    document.__enter__.return_value = document
    document.__iter__.side_effect = RuntimeError('破損PDF')
    mocker.patch('service.barcode_reader.pymupdf.open', return_value=document)

    with pytest.raises(RuntimeError):
        extract_images_from_pdf('broken.pdf')

    document.__exit__.assert_called_once()


def test_extract_images_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        extract_images_from_pdf(str(tmp_path / 'missing.pdf'))


# --- _decode_code128（P0） ---


def test_decode_returns_value_found_on_first_attempt(mocker: MockerFixture) -> None:
    decode = mocker.patch(
        'service.barcode_reader.decode', return_value=[fake_barcode(b'ABC123', 10, 10)]
    )
    denoise = mocker.patch('service.barcode_reader.cv2.fastNlMeansDenoising')

    assert _decode_code128(np.zeros((4, 4), dtype=np.uint8)) == 'ABC123'
    assert decode.call_count == 1
    denoise.assert_not_called()


def test_decode_retries_with_binarized_image(mocker: MockerFixture) -> None:
    gray = np.zeros((4, 4), dtype=np.uint8)
    binarized = np.ones((4, 4), dtype=np.uint8)
    decode = mocker.patch(
        'service.barcode_reader.decode', side_effect=[[], [fake_barcode(b'RETRY', 0, 0)]]
    )
    denoise = mocker.patch('service.barcode_reader.cv2.fastNlMeansDenoising', return_value=gray)
    threshold = mocker.patch('service.barcode_reader.cv2.threshold', return_value=(0, binarized))

    assert _decode_code128(gray) == 'RETRY'
    denoise.assert_called_once()
    threshold.assert_called_once()
    assert decode.call_args_list[1].args[0] is binarized


def test_decode_returns_none_when_both_attempts_fail(mocker: MockerFixture) -> None:
    mocker.patch('service.barcode_reader.decode', return_value=[])
    mocker.patch('service.barcode_reader.cv2.fastNlMeansDenoising', return_value=np.zeros((4, 4)))
    mocker.patch('service.barcode_reader.cv2.threshold', return_value=(0, np.zeros((4, 4))))

    assert _decode_code128(np.zeros((4, 4), dtype=np.uint8)) is None


def test_decode_selects_top_left_barcode(mocker: MockerFixture) -> None:
    mocker.patch('service.barcode_reader.decode', return_value=[
        fake_barcode(b'BOTTOM', 500, 10),
        fake_barcode(b'TOPLEFT', 10, 20),
        fake_barcode(b'RIGHT', 20, 400),
    ])

    assert _decode_code128(np.zeros((4, 4), dtype=np.uint8)) == 'TOPLEFT'


def test_decode_keeps_first_barcode_when_positions_tie(mocker: MockerFixture) -> None:
    mocker.patch('service.barcode_reader.decode', return_value=[
        fake_barcode(b'FIRST', 10, 20),
        fake_barcode(b'SECOND', 20, 10),
    ])

    assert _decode_code128(np.zeros((4, 4), dtype=np.uint8)) == 'FIRST'


def test_decode_raises_for_non_utf8_payload(mocker: MockerFixture) -> None:
    mocker.patch(
        'service.barcode_reader.decode', return_value=[fake_barcode(b'\xff\xfe', 0, 0)]
    )

    with pytest.raises(UnicodeDecodeError):
        _decode_code128(np.zeros((4, 4), dtype=np.uint8))


# --- read_barcode_from_pdf（P0） ---


@pytest.fixture
def two_images(mocker: MockerFixture) -> None:
    mocker.patch(
        'service.barcode_reader.extract_images_from_pdf',
        return_value=[Image.new('L', (4, 4)), Image.new('L', (4, 4))],
    )


def test_read_barcode_stops_at_first_hit(two_images: None, mocker: MockerFixture) -> None:
    decode = mocker.patch('service.barcode_reader._decode_code128', return_value='ABC123')

    assert read_barcode_from_pdf('any.pdf') == 'ABC123'
    assert decode.call_count == 1


def test_read_barcode_continues_to_next_image(two_images: None, mocker: MockerFixture) -> None:
    decode = mocker.patch('service.barcode_reader._decode_code128', side_effect=[None, 'SECOND'])

    assert read_barcode_from_pdf('any.pdf') == 'SECOND'
    assert decode.call_count == 2


def test_read_barcode_returns_none_when_nothing_found(
    two_images: None,
    mocker: MockerFixture,
) -> None:
    mocker.patch('service.barcode_reader._decode_code128', return_value=None)

    assert read_barcode_from_pdf('any.pdf') is None


def test_read_barcode_recovers_from_decode_error(
    two_images: None,
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
) -> None:
    mocker.patch(
        'service.barcode_reader._decode_code128',
        side_effect=[UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid'), 'SECOND'],
    )

    assert read_barcode_from_pdf('any.pdf') == 'SECOND'
    assert caplog.records[0].levelno == logging.WARNING


def test_read_barcode_returns_none_for_pdf_without_images(mocker: MockerFixture) -> None:
    mocker.patch('service.barcode_reader.extract_images_from_pdf', return_value=[])

    assert read_barcode_from_pdf('any.pdf') is None


def test_read_barcode_propagates_extraction_error(mocker: MockerFixture) -> None:
    """PDF展開の失敗はprocess_pdf側でエラーフォルダ行きとして扱う"""
    mocker.patch(
        'service.barcode_reader.extract_images_from_pdf', side_effect=RuntimeError('破損PDF')
    )

    with pytest.raises(RuntimeError):
        read_barcode_from_pdf('any.pdf')
