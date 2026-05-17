"""
EasyOCR ашиглан зурган дэх текстийг гарган авах модуль.
Монгол ('mn') болон орос ('ru') кирилл үсгийг дэмжинэ.
Vision LLM-ээс ~3–10 дахин хурдан.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

CROPS_DIR = Path(__file__).parent / "data" / "crops"
MANIFEST   = Path(__file__).parent / "data" / "manifest.jsonl"

if TYPE_CHECKING:
    import easyocr as _easyocr_module

_reader: "_easyocr_module.Reader | None" = None


def _get_reader():
    """Reader-г нэг удаа ачаалж дахин ашиглана."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["mn", "ru", "en"], verbose=False)
    return _reader


def extract_texts(image_path: str | Path) -> list[dict]:
    """
    Зургаас бүх текстийн мөрийг гарган авна.

    Returns:
        [{"text": "...", "confidence": 0.95, "area": 1234}, ...]
        area-гаар буурах эрэмбэтэй — том текст (гарчиг) эхэнд байна.
    """
    try:
        img = np.array(Image.open(image_path).convert("RGB"))
        reader = _get_reader()
        raw = reader.readtext(img, detail=1, paragraph=False)
    except Exception as e:
        return [{"error": str(e)}]

    results = []
    for bbox, text, conf in raw:
        text = text.strip()
        if not text or conf < 0.3:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        results.append({
            "text":       text,
            "confidence": round(conf, 3),
            "area":       int(area),
            "bbox":       bbox,      # crop хадгалахад хэрэгтэй
        })

    results.sort(key=lambda x: x["area"], reverse=True)
    return results


def save_crops(
    image_path: str | Path,
    ocr_results: list[dict],
    corrected_texts: dict[int, str] | None = None,
) -> int:
    """
    OCR-ийн bbox бүрийг хэрчиж crop болгон хадгална.
    manifest.jsonl-д (crop_path, ocr_text, corrected_text, confidence) бичнэ.

    Args:
        image_path:      Эх зургийн зам
        ocr_results:     extract_texts()-ийн үр дүн (bbox агуулсан)
        corrected_texts: {index: "засварласан текст"} — хэрэглэгч засварласан бол

    Returns:
        Хадгалсан crop-ийн тоо
    """
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved = 0

    for i, item in enumerate(ocr_results):
        bbox = item.get("bbox")
        if not bbox:
            continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
        x2, y2 = int(max(xs)), int(max(ys))
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img.crop((x1, y1, x2, y2))
        crop_name = f"{stamp}_{i:03d}.jpg"
        crop_path = CROPS_DIR / crop_name
        crop.save(crop_path, "JPEG", quality=95)

        corrected = (corrected_texts or {}).get(i)
        record = {
            "crop":       str(crop_path),
            "ocr_text":   item["text"],
            "corrected":  corrected,          # None бол засвар байхгүй
            "confidence": item["confidence"],
            "source":     str(image_path),
            "timestamp":  stamp,
        }
        with MANIFEST.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        saved += 1

    return saved


def texts_to_book_info(texts: list[dict]) -> dict:
    """
    OCR-ийн текст жагсаалтаас гарчиг/зохиогчийг таамаглана.
    Хамгийн том (area) текст → гарчиг
    2-р том → зохиогч (хэрэв хүний нэр мэт харагдвал)
    """
    if not texts or texts[0].get("error"):
        return {"title": None, "author": None, "confidence": "low",
                "error": texts[0].get("error") if texts else "Текст олдсонгүй"}

    plain = [t["text"] for t in texts]
    title = plain[0] if plain else None
    author = None

    # 2–4-р мөрөөс зохиогч хайх — богино, цэг/таслал агуулсан байвал зохиогч байж болно
    for t in plain[1:4]:
        if 2 <= len(t.split()) <= 5 and not t.isupper():
            author = t
            break

    avg_conf = sum(t["confidence"] for t in texts) / len(texts)
    confidence = "high" if avg_conf >= 0.85 else "medium" if avg_conf >= 0.6 else "low"

    return {
        "title":      title,
        "author":     author,
        "language":   _detect_language(plain),
        "publisher":  None,
        "confidence": confidence,
        "ocr_lines":  plain,
    }


def _detect_language(texts: list[str]) -> str | None:
    """Текстийн дийлэнх хэсгийн кодчиллоор хэлийг таамаглана."""
    joined = " ".join(texts)
    cyrillic = sum(1 for c in joined if "\u0400" <= c <= "\u04ff")
    latin    = sum(1 for c in joined if c.isascii() and c.isalpha())
    if cyrillic > latin:
        # Өө Үү байвал Монгол, эс бөгөөс орос гэж таамаглана
        mongolian_chars = sum(1 for c in joined if c in "ӨөҮү")
        return "mn" if mongolian_chars > 0 else "ru"
    return "en" if latin > 0 else None
