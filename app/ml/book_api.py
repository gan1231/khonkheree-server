"""
Номын мэдээллийг олон нийтийн API-аас авч баяжуулах модуль.
Open Library (үнэгүй, түлхүүр шаардахгүй) болон Google Books-г ашиглана.
"""

import urllib.parse
from typing import Any

import requests


OPEN_LIBRARY_BASE = "https://openlibrary.org"
GOOGLE_BOOKS_BASE = "https://www.googleapis.com/books/v1/volumes"
TIMEOUT = 8  # секунд

# Зарим API (ялангуяа Open Library) User-Agent шаарддаг
HEADERS = {
    "User-Agent": "BookAgent/1.0 (local AI book identifier)",
    "Accept": "application/json",
}


def enrich_by_isbn(isbn: str) -> dict[str, Any]:
    """ISBN дугаараар номын мэдээлэл авах."""
    # Open Library-аас эхлээд үзнэ
    data = _open_library_by_isbn(isbn)
    if data:
        return data

    # Google Books-с туршиж үзнэ
    return _google_books_search(f"isbn:{isbn}")


def enrich_by_title(title: str, author: str | None = None) -> dict[str, Any]:
    """Гарчиг, зохиогчоор хайж номын мэдээлэл авах."""
    query_parts = [title]
    if author:
        query_parts.append(author)
    query = " ".join(query_parts)

    # Open Library search хамгийн сайн ажилладаг
    data = _open_library_search(title, author)
    if data:
        return data

    # Хоосон байвал Google Books
    return _google_books_search(query)


def _open_library_by_isbn(isbn: str) -> dict[str, Any] | None:
    """Open Library-аас ISBN-оор хайх."""
    url = f"{OPEN_LIBRARY_BASE}/api/books"
    params = {
        "bibkeys": f"ISBN:{isbn}",
        "format": "json",
        "jscmd": "data",
    }

    try:
        r = requests.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
        r.raise_for_status()
        result = r.json()
    except Exception as e:
        return {"error": f"Open Library ISBN хайлт амжилтгүй: {e}"}

    key = f"ISBN:{isbn}"
    if key not in result:
        return None

    book = result[key]
    return {
        "source": "open_library",
        "isbn": isbn,
        "title": book.get("title"),
        "subtitle": book.get("subtitle"),
        "authors": [a.get("name") for a in book.get("authors", [])],
        "publish_date": book.get("publish_date"),
        "publishers": [p.get("name") for p in book.get("publishers", [])],
        "number_of_pages": book.get("number_of_pages"),
        "subjects": [s.get("name") for s in book.get("subjects", [])][:5],
        "cover_url": (book.get("cover") or {}).get("large")
        or (book.get("cover") or {}).get("medium"),
        "description": _extract_description(book),
        "url": book.get("url"),
    }


def _open_library_search(title: str, author: str | None = None) -> dict[str, Any] | None:
    """Open Library search endpoint ашиглан хайх."""
    params = {"title": title, "limit": 1}
    if author:
        params["author"] = author

    try:
        r = requests.get(f"{OPEN_LIBRARY_BASE}/search.json",
            params=params,
            timeout=TIMEOUT, headers=HEADERS,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": f"Open Library search амжилтгүй: {e}"}

    docs = data.get("docs", [])
    if not docs:
        return None

    doc = docs[0]
    cover_id = doc.get("cover_i")
    cover_url = (
        f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg" if cover_id else None
    )

    return {
        "source": "open_library_search",
        "title": doc.get("title"),
        "authors": doc.get("author_name", []),
        "first_publish_year": doc.get("first_publish_year"),
        "publishers": doc.get("publisher", [])[:3],
        "isbn": (doc.get("isbn") or [None])[0],
        "number_of_pages": doc.get("number_of_pages_median"),
        "subjects": doc.get("subject", [])[:5],
        "cover_url": cover_url,
        "url": f"{OPEN_LIBRARY_BASE}{doc.get('key', '')}" if doc.get("key") else None,
    }


def _google_books_search(query: str) -> dict[str, Any]:
    """Google Books API-аар хайх (түлхүүргүй, хязгаартай)."""
    params = {"q": query, "maxResults": 1}

    try:
        r = requests.get(GOOGLE_BOOKS_BASE, params=params, timeout=TIMEOUT, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": f"Google Books хайлт амжилтгүй: {e}"}

    items = data.get("items", [])
    if not items:
        return {"error": "Ямар ч ном олдсонгүй"}

    info = items[0].get("volumeInfo", {})
    return {
        "source": "google_books",
        "title": info.get("title"),
        "subtitle": info.get("subtitle"),
        "authors": info.get("authors", []),
        "publisher": info.get("publisher"),
        "published_date": info.get("publishedDate"),
        "description": info.get("description"),
        "page_count": info.get("pageCount"),
        "categories": info.get("categories", []),
        "average_rating": info.get("averageRating"),
        "language": info.get("language"),
        "cover_url": (info.get("imageLinks") or {}).get("thumbnail"),
        "url": info.get("infoLink"),
    }


def _extract_description(book: dict) -> str | None:
    """Open Library-ийн эмх замбараагүй description талбарыг задлах."""
    desc = book.get("description") or book.get("notes")
    if isinstance(desc, dict):
        return desc.get("value")
    return desc


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Хэрэглээ:")
        print("  python book_api.py isbn <isbn>")
        print("  python book_api.py title <гарчиг> [зохиогч]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "isbn":
        result = enrich_by_isbn(sys.argv[2])
    elif mode == "title":
        author = sys.argv[3] if len(sys.argv) > 3 else None
        result = enrich_by_title(sys.argv[2], author)
    else:
        print(f"Тодорхойгүй горим: {mode}")
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))
