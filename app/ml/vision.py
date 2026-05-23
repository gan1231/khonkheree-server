"""
Vision загвартай харилцах модуль.
Ollama-аар дамжуулан локал vision загвар (qwen2.5vl, llama3.2-vision г.м.)
ашиглан номын хавтаснаас гарчиг, зохиогчийг гарган авна.
"""

import json
import re
from pathlib import Path

import ollama


DEFAULT_MODEL = "moondream:latest"

PROMPT_JSON = """You are reading a book cover. Transcribe every text you see exactly as written, then return ONLY this JSON — no explanation, no markdown fences:

{"title":"...","author":"...","language":"mn|en|ru|zh|ja|ko|de|fr or null","publisher":"...","confidence":"low|medium|high"}

Rules:
- Copy text verbatim from the image, do not translate.
- language is the language of the book text, not your response language.
- Use null for fields you cannot read."""

# moondream зэрэг JSON дагаж чаддахгүй жижиг загваруудад зориулсан энгийн prompt
PROMPT_PLAIN = "Read this image and output only the visible text, exactly as written. No coordinates, no bounding boxes, no descriptions — only the words you see."

# JSON prompt дагадаггүй мэдэгдэж байгаа загварууд
_PLAIN_TEXT_MODELS = {"moondream", "moondream:latest", "moondream2"}

# Хуучин нэр нийцтэй байдлаар хадгална
PROMPT = PROMPT_JSON


def extract_book_info(
    image_path: str | Path,
    model: str = DEFAULT_MODEL,
    host: str | None = None,
) -> dict:
    """
    Номын хавтасны зургаас мэдээлэл гаргана.

    Урсгал:
      1. Donut — Хэрэв идэвхтэй бол хамгийн түрүүнд Donut моделиор уншина
      2. EasyOCR — Кирилл/латин текстийг хурдан гарган авна (Donut-ийн fallback)
      3. OCR амжилттай бол Vision LLM-д OCR текстийг дамжуулж бүтэцжүүлнэ
      4. OCR бүтэлгүйтвэл Vision LLM зургийг шууд уншина (fallback)
    """
    from app.core.config import settings
    
    # 1-р алхам: Donut ажиллуулах (хэрэв идэвхтэй бол)
    if getattr(settings, "USE_DONUT", False):
        try:
            from app.ml.donut_model import get_donut_predictor
            predictor = get_donut_predictor(settings.DONUT_MODEL_PATH)
            donut_result = predictor.predict(str(image_path))
            
            # Хэрэв өндөр эсвэл дунд итгэлцүүртэй таньсан бол шууд буцаана
            if donut_result.get("confidence") in ("high", "medium") and donut_result.get("title"):
                return donut_result
        except Exception:
            # Алдаа гарвал дараагийн OCR / Ollama урсгал руу fallback хийнэ
            pass

    from ocr import extract_texts, texts_to_book_info, save_crops

    # 2-р алхам: EasyOCR (fallback)
    ocr_results = extract_texts(image_path)
    ocr_ok = ocr_results and not ocr_results[0].get("error")

    # Crop хадгалах — сургалтын өгөгдөл автоматаар хуримтлагдана
    if ocr_ok:
        save_crops(image_path, ocr_results)

    if ocr_ok:
        ocr_info = texts_to_book_info(ocr_results)
        ocr_lines = ocr_info.get("ocr_lines", [])

        # OCR текстийг Vision LLM-д дамжуулж бүтэцжүүлнэ (зураг биш)
        if ocr_lines:
            structured = _structure_with_llm(ocr_lines, model, host)
            if structured.get("title"):
                structured["ocr_lines"] = ocr_lines
                structured["raw"] = "\n".join(ocr_lines)
                return structured

        # LLM бүтэцжүүлж чадаагүй бол OCR-ийн шууд таамаглалыг ашиглана
        if ocr_info.get("title"):
            ocr_info["raw"] = "\n".join(ocr_lines)
            return ocr_info

    # 2-р үе шат: Vision LLM fallback — зургийг шууд уншина
    return _vision_llm(image_path, model, host)


def _structure_with_llm(ocr_lines: list[str], model: str, host: str | None) -> dict:
    """OCR текстийг Ollama-д дамжуулж JSON бүтэцтэй болгоно (vision биш)."""
    client = ollama.Client(host=host) if host else ollama
    text_block = "\n".join(ocr_lines[:15])

    prompt = f"""The following lines of text were extracted from a book cover using OCR:

{text_block}

Return ONLY this JSON — no explanation, no markdown:
{{"title":"...","author":"...","language":"mn|en|ru|zh|ja|ko or null","publisher":"...","confidence":"low|medium|high"}}

Rules: copy text verbatim, use null for unknown fields."""

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
    except Exception:
        return {"title": None}

    return _parse_json_response(response["message"]["content"])


def _vision_llm(image_path: str | Path, model: str, host: str | None) -> dict:
    """Vision LLM-аар зургийг шууд уншина (OCR-ийн fallback)."""
    client = ollama.Client(host=host) if host else ollama

    model_key = model.split(":")[0].lower() if model else ""
    use_plain = model in _PLAIN_TEXT_MODELS or model_key in _PLAIN_TEXT_MODELS
    prompt = PROMPT_PLAIN if use_plain else PROMPT_JSON

    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
            options={"temperature": 0.1},
        )
    except Exception as e:
        return {
            "error": f"Vision загвар дуудахад алдаа гарлаа: {e}",
            "title": None,
            "author": None,
            "confidence": "low",
        }

    raw_text = response["message"]["content"]
    return _parse_json_response(raw_text)


def _parse_json_response(text: str) -> dict:
    """
    Загварын хариунаас JSON-г гарган авна.
    JSON олдохгүй бол plain text-ээс талбаруудыг ялгах fallback ажиллана.
    """
    # ```json ... ``` блок хасах
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.replace("```", "").strip()

    # Хамгийн гадна талын { ... } -г олох
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
            for key in ("title", "author", "language", "publisher", "confidence"):
                data.setdefault(key, None)
            return data
        except json.JSONDecodeError:
            pass

    # JSON олдоогүй — plain text-ээс талбар ялгах fallback
    return _parse_plain_text(text)


def _parse_plain_text(text: str) -> dict:
    """
    JSON биш plain text хариунаас гарчиг, зохиогч зэргийг ялган авна.
    moondream зэрэг жижиг загваруудад зориулсан fallback.
    """
    result = {
        "title": None,
        "author": None,
        "language": None,
        "publisher": None,
        "confidence": "low",
        "raw": text[:300],
    }

    lower = text.lower()

    # "Title: ..." / "Book: ..." гэх тодорхой хэлбэр хайх
    patterns = {
        "title":     r"(?:title|book(?:\s+title)?|гарчиг)\s*[:：]\s*(.+)",
        "author":    r"(?:author|by|зохиогч|зохиол[чч])\s*[:：]\s*(.+)",
        "language":  r"(?:language|хэл)\s*[:：]\s*(\w+)",
        "publisher": r"(?:publisher|published\s+by|хэвлэл)\s*[:：]\s*(.+)",
    }

    for field, pattern in patterns.items():
        m = re.search(pattern, lower)
        if m:
            result[field] = text[m.start(1):m.end(1)].strip().strip('"\'')

    # Хэрэв ямар ч хэлбэр олдоогүй бол эхний мөрийг гарчиг гэж үз
    if not result["title"]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for candidate in lines:
            candidate = candidate.strip('"\'').strip()
            if _is_coordinate(candidate):
                continue
            if len(candidate) < 120 and "error" not in candidate.lower():
                result["title"] = candidate
                result["confidence"] = "low"
                break

    return result


def _is_coordinate(text: str) -> bool:
    """Загварын хариу bounding box координат мөн эсэхийг шалгана."""
    # [0.39, 0.13, 0.99, 0.3] гэх хэлбэр
    return bool(re.fullmatch(r"\[[\s\d.,]+\]", text.strip()))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Хэрэглээ: python vision.py <зургийн_зам> [загварын_нэр]")
        sys.exit(1)

    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    result = extract_book_info(sys.argv[1], model=model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
