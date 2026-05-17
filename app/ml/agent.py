"""
Номын агентын үндсэн логик.

Урсгал:
  1. ISBN бар код хайх (хамгийн найдвартай)
  2. Олдоогүй бол vision загвараар хавтсыг уншуулах
  3. Аль нэгийг ашиглан API-аас дэлгэрэнгүй мэдээлэл авах
  4. Нэгтгэсэн үр дүнг буцаах
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from isbn_scanner import scan_isbn
from vision import DEFAULT_MODEL, extract_book_info
from book_api import enrich_by_isbn, enrich_by_title
from db import BookDatabase


@dataclass
class BookResult:
    """Агентын эцсийн гаргалт."""

    method: str  # "isbn" эсвэл "vision"
    success: bool
    isbn: str | None = None
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    publisher: str | None = None
    published: str | None = None
    language: str | None = None
    description: str | None = None
    cover_url: str | None = None
    page_count: int | None = None
    categories: list[str] = field(default_factory=list)
    rating: float | None = None
    source: str | None = None
    confidence: str | None = None
    note: str | None = None
    raw_vision: dict | None = None
    db_id: int | None = None  # DB-ээс олдсон үед existing record-ийн id

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {})}


class BookAgent:
    """Номын таних агент."""

    def __init__(
        self,
        vision_model: str = DEFAULT_MODEL,
        ollama_host: str | None = None,
        db: BookDatabase | None = None,
    ):
        self.vision_model = vision_model
        self.ollama_host = ollama_host
        self.db = db

    def identify(self, image_path: str | Path) -> BookResult:
        """
        Зургаас ном таних гол функц.

        Урсгал:
          1. ISBN бар код → API (хамгийн найдвартай, хурдан)
          2. Vision → зургаас текст гарган авах
          3. Текстээр DB-д хайх
          4. Текстээр гадаад API-д хайх
          5. Текст таарахгүй бол CLIP зургийн ижилтэлээр DB-д хайх (fallback)
        """
        image_path = Path(image_path)
        if not image_path.exists():
            return BookResult(
                method="none",
                success=False,
                note=f"Зураг олдсонгүй: {image_path}",
            )

        # 1-р алхам: ISBN бар код — хурдан, найдвартай
        isbn = scan_isbn(image_path)
        if isbn:
            result = self._via_isbn(isbn)
            if result.success:
                return result

        # 2-р алхам: Vision — зургаас текст гарган авах
        vision_data = extract_book_info(
            image_path,
            model=self.vision_model,
            host=self.ollama_host,
        )

        title  = vision_data.get("title")
        author = vision_data.get("author")
        language   = vision_data.get("language")
        confidence = vision_data.get("confidence")

        # 3-р алхам: Текстээр DB-д хайх
        if title and self.db:
            db_row = self.db.search_by_text(title, author)
            if db_row:
                return BookResult(
                    method="db",
                    success=True,
                    isbn=db_row.get("isbn"),
                    title=db_row.get("title"),
                    authors=db_row.get("authors") or [],
                    publisher=db_row.get("publisher"),
                    published=db_row.get("published"),
                    language=db_row.get("language"),
                    description=db_row.get("description"),
                    cover_url=db_row.get("cover_url"),
                    page_count=db_row.get("page_count"),
                    categories=db_row.get("categories") or [],
                    rating=db_row.get("rating"),
                    source=db_row.get("source"),
                    confidence=db_row.get("confidence"),
                    note=f"Мэдээллийн сангаас текстийн хайлтаар олдлоо «{title}»",
                    db_id=db_row.get("id"),
                )

        # 4-р алхам: Текстээр гадаад API-д хайх
        if title:
            result = self._via_vision_with_api(vision_data, image_path)
            if result.success:
                return result

        # 5-р алхам: Текст таарахгүй → CLIP зургийн ижилтэлээр DB fallback
        if self.db:
            db_row, similarity = self.db.search_by_image(image_path)
            if db_row:
                return BookResult(
                    method="db",
                    success=True,
                    isbn=db_row.get("isbn"),
                    title=db_row.get("title"),
                    authors=db_row.get("authors") or [],
                    publisher=db_row.get("publisher"),
                    published=db_row.get("published"),
                    language=db_row.get("language"),
                    description=db_row.get("description"),
                    cover_url=db_row.get("cover_url"),
                    page_count=db_row.get("page_count"),
                    categories=db_row.get("categories") or [],
                    rating=db_row.get("rating"),
                    source=db_row.get("source"),
                    confidence=db_row.get("confidence"),
                    note=f"Мэдээллийн сангаас зургийн ижилтэлээр олдлоо ({similarity:.1%})",
                    db_id=db_row.get("id"),
                )

        # Бүх замнал амжилтгүй — vision мэдээллийг л буцаана
        if vision_data.get("error"):
            return BookResult(
                method="vision",
                success=False,
                note=vision_data["error"],
                raw_vision=vision_data,
            )

        return BookResult(
            method="vision",
            success=bool(title),
            title=title,
            authors=[author] if author else [],
            language=language,
            publisher=vision_data.get("publisher"),
            confidence=confidence,
            note="Зөвхөн хавтасны мэдээлэл — API болон DB-аас олдсонгүй",
            raw_vision=vision_data,
        )

    def _via_isbn(self, isbn: str) -> BookResult:
        """ISBN олдсон үеийн замнал."""
        data = enrich_by_isbn(isbn)

        if not data or data.get("error"):
            return BookResult(
                method="isbn",
                success=False,
                isbn=isbn,
                note=data.get("error") if data else "API хариу буцаасангүй",
            )

        return self._build_result(data, method="isbn", isbn=isbn)

    def _via_vision_with_api(self, vision_data: dict, image_path: Path) -> BookResult:
        """Vision-ээр гарган авсан текстийг гадаад API-аар баяжуулах замнал."""
        title      = vision_data.get("title")
        author     = vision_data.get("author")
        language   = vision_data.get("language")
        confidence = vision_data.get("confidence")

        # Монгол ном — олон улсын API-д байхгүй тул шууд буцаана
        if language == "mn":
            return BookResult(
                method="vision",
                success=True,
                title=title,
                authors=[author] if author else [],
                language=language,
                publisher=vision_data.get("publisher"),
                confidence=confidence,
                note="Монгол ном — олон улсын мэдээллийн сангаас олдоогүй, зөвхөн хавтасны мэдээлэл",
                raw_vision=vision_data,
            )

        api_data = enrich_by_title(title, author)

        if not api_data or api_data.get("error"):
            return BookResult(
                method="vision",
                success=False,
                title=title,
                authors=[author] if author else [],
                language=language,
                confidence=confidence,
                note=(api_data or {}).get("error", "API-аас мэдээлэл авагдсангүй"),
                raw_vision=vision_data,
            )

        result = self._build_result(api_data, method="vision")
        result.confidence = confidence
        result.raw_vision = vision_data
        return result

    @staticmethod
    def _build_result(
        data: dict,
        method: str,
        isbn: str | None = None,
    ) -> BookResult:
        """API-ийн хариуг BookResult болгон хөрвүүлэх."""
        # Authors - хэрэв зөвхөн publisher талбар бол түүнийг ашиглана
        authors = data.get("authors") or []
        publisher = data.get("publisher") or (
            (data.get("publishers") or [None])[0]
        )

        published = (
            data.get("published_date")
            or data.get("publish_date")
            or str(data.get("first_publish_year") or "") or None
        )

        return BookResult(
            method=method,
            success=True,
            isbn=isbn or data.get("isbn"),
            title=data.get("title"),
            authors=authors if isinstance(authors, list) else [authors],
            publisher=publisher,
            published=published,
            language=data.get("language"),
            description=data.get("description"),
            cover_url=data.get("cover_url"),
            page_count=data.get("page_count") or data.get("number_of_pages"),
            categories=data.get("categories") or data.get("subjects") or [],
            rating=data.get("average_rating"),
            source=data.get("source"),
        )


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Хэрэглээ: python agent.py <зургийн_зам> [загвар]")
        sys.exit(1)

    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    agent = BookAgent(vision_model=model)
    result = agent.identify(sys.argv[1])

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
