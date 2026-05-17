"""
ISBN бар код танигч.
Номын хавтас дээрх бар кодыг уншаад ISBN дугаарыг буцаана.
"""

from pathlib import Path
from PIL import Image, ImageOps
from pyzbar.pyzbar import decode, ZBarSymbol


def scan_isbn(image_path: str | Path) -> str | None:
    """
    Зургаас ISBN бар код хайна.

    Args:
        image_path: Зургийн файлын зам

    Returns:
        ISBN-10 эсвэл ISBN-13 дугаар, эсвэл олдоогүй бол None
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"[ISBN] Зураг нээж чадсангүй: {e}")
        return None

    # Эхлээд шууд уншиж үзнэ
    isbn = _try_decode(img)
    if isbn:
        return isbn

    # Олдоогүй бол саарал болгоод, контраст тохируулж дахин оролдоно
    gray = ImageOps.grayscale(img)
    isbn = _try_decode(gray)
    if isbn:
        return isbn

    # Сүүлийн оролдлого: автоконтраст
    enhanced = ImageOps.autocontrast(gray)
    return _try_decode(enhanced)


def _try_decode(img: Image.Image) -> str | None:
    """PIL зураг дотроос EAN13/ISBN бар код хайна."""
    barcodes = decode(img, symbols=[ZBarSymbol.EAN13, ZBarSymbol.EAN8])

    for b in barcodes:
        code = b.data.decode("utf-8")
        # ISBN-13 нь 978 эсвэл 979-ээр эхэлдэг
        if code.startswith(("978", "979")) and len(code) == 13:
            return code

    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Хэрэглээ: python isbn_scanner.py <зургийн_зам>")
        sys.exit(1)

    result = scan_isbn(sys.argv[1])
    if result:
        print(f"ISBN олдлоо: {result}")
    else:
        print("ISBN олдсонгүй")
