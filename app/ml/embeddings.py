"""
CLIP зургийн embedding тооцоолох модуль (Алхам 2 — Vector DB).

sentence-transformers-ийн clip-ViT-B-32 загварыг ашиглан номын
хавтасны зургийг 512 хэмжээст векторт хөрвүүлнэ.

Хадгалсан embedding-ийг pgvector cosine_distance ашиглан хайхад:
  - Зураг дахин upload хийхэд тэр дороо ижил ном олдоно
  - ISBN эсвэл текст таних амжилтгүй болсон үед зургийн ижил хайлт ажиллана
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_clip_model = None


def _get_clip_model():
    """CLIP моделийн Singleton жишээг буцаана (анхны дуудалтад л ачаалагдана)."""
    global _clip_model
    if _clip_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("CLIP clip-ViT-B-32 загварыг ачаалж байна...")
        _clip_model = SentenceTransformer("clip-ViT-B-32")
        logger.info("CLIP загвар бэлэн боллоо.")
    return _clip_model


async def compute_from_path(image_path: str | Path) -> Optional[list[float]]:
    """
    Локал зургийн замаар CLIP embedding тооцоолно.
    Asyncio executor ашиглан CPU-bound ажлыг event loop-ийг блоклохгүйгээр явуулна.
    """
    import asyncio
    from PIL import Image

    try:
        model = _get_clip_model()
        img = Image.open(str(image_path)).convert("RGB")

        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, lambda: model.encode(img))
        return vec.tolist()
    except Exception as exc:
        logger.warning("CLIP embedding (path) тооцоолоход алдаа: %s", exc)
        return None


async def compute_from_url(url: str) -> Optional[list[float]]:
    """
    Зургийн URL-аас татаж CLIP embedding тооцоолно.
    cover_url нь R2/CDN дээр байрших тохиолдолд ашиглана.
    """
    import asyncio
    import httpx
    from PIL import Image

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")

        model = _get_clip_model()
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, lambda: model.encode(img))
        return vec.tolist()
    except Exception as exc:
        logger.warning("CLIP embedding (url) тооцоолоход алдаа: %s", exc)
        return None


async def compute_from_bytes(data: bytes) -> Optional[list[float]]:
    """Raw bytes-аас embedding тооцоолно (identify-book endpoint-д ашиглана)."""
    import asyncio
    from PIL import Image

    try:
        model = _get_clip_model()
        img = Image.open(io.BytesIO(data)).convert("RGB")

        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, lambda: model.encode(img))
        return vec.tolist()
    except Exception as exc:
        logger.warning("CLIP embedding (bytes) тооцоолоход алдаа: %s", exc)
        return None
