from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import tempfile, os

router = APIRouter(prefix="/ml", tags=["ml"])


class BookIdentifyResult(BaseModel):
    isbn: Optional[str]
    title: Optional[str]
    author: Optional[str]
    synopsis: Optional[str]
    cover_url: Optional[str]
    source: str  # "barcode" | "vision" | "api"
    confidence: float


@router.post("/identify-book", response_model=BookIdentifyResult)
async def identify_book(image: UploadFile = File(...)):
    """
    Номын зураг хүлээн авч ISBN / гарчиг / зохиогч мэдээллийг буцаана.
    agent800-ийн BookAgent ашиглана.
    """
    try:
        from app.ml.agent import BookAgent
    except ImportError:
        raise HTTPException(status_code=503, detail="ML модуль ачаалагдаагүй байна")

    suffix = os.path.splitext(image.filename or "img.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        agent = BookAgent()
        result = agent.identify(tmp_path)
    finally:
        os.unlink(tmp_path)

    return BookIdentifyResult(
        isbn=result.get("isbn"),
        title=result.get("title"),
        author=result.get("author"),
        synopsis=result.get("synopsis"),
        cover_url=result.get("cover_url"),
        source=result.get("source", "vision"),
        confidence=result.get("confidence", 0.0),
    )
