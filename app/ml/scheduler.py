"""
Автомат дахин сургалтын хуваарьт ажил (Алхам 4 — Auto Retrain).

APScheduler AsyncIOScheduler ашиглан FastAPI-тай нэгдмэл ажиллана.
Шөнийн тогтсон цагт шалгаж, шинэ батлагдсан датасет хангалттай бол
Donut моделийг автоматаар дахин сургана.

Урсгал:
  1. [RETRAIN_CRON_HOUR] цагт сэрэх
  2. training_samples WHERE used_in_training=False тоолох
  3. >= AUTO_RETRAIN_THRESHOLD → export + train + reload
  4. Ашигласан бүртгэлүүдийг used_in_training=True болгох
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_scheduler = None


def get_scheduler():
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="Asia/Ulaanbaatar")
    return _scheduler


# ─── Дотоод ажлын функцууд ───────────────────────────────────────────────────

async def _check_and_retrain() -> None:
    """Шинэ датасет хангалттай бол автомат дахин сургалт явуулна."""
    from sqlalchemy import select, func
    from app.db.session import AsyncSessionLocal
    from app.db.models import TrainingSample
    from app.core.config import settings

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(func.count(TrainingSample.id)).where(
                    TrainingSample.used_in_training.is_(False)
                )
            )
            pending = result.scalar_one()

        logger.info("Автомат шалгалт: сургалтанд ашиглагдаагүй датасет = %d", pending)

        if pending < settings.AUTO_RETRAIN_THRESHOLD:
            logger.info(
                "Босго (%d) хүрсэнгүй (%d). Дараагийн шалгалтыг хүлээнэ.",
                settings.AUTO_RETRAIN_THRESHOLD,
                pending,
            )
            return

        logger.info("Босго хүрлээ — дахин сургалт эхлүүлж байна...")
        await _run_retrain()

    except Exception as exc:
        logger.error("Автомат шалгалтад алдаа гарлаа: %s", exc)


async def _run_retrain() -> None:
    """Датасет экспортлож, Donut-ийг сургаж, Singleton-ийг цэвэрлэнэ."""
    import asyncio
    from sqlalchemy import update
    from app.db.session import AsyncSessionLocal
    from app.db.models import TrainingSample
    from app.core.config import settings
    from app.ml.train_donut import export_db_dataset, train_model

    data_dir = "donut_dataset_auto"

    try:
        # 1. PostgreSQL-ийн баталгаажсан мэдээллээс датасет экспортлох
        logger.info("Датасет экспортлож байна → %s/", data_dir)
        await export_db_dataset(data_dir)

        # 2. Синхрон сургалтыг thread executor-т явуулах (event loop блоклохгүй)
        logger.info("Donut загварыг сургаж байна...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: train_model(
                dataset_dir=data_dir,
                model_path=settings.DONUT_MODEL_PATH,
                output_model_dir=settings.DONUT_OUTPUT_MODEL_DIR,
                epochs=3,
            ),
        )

        # 3. Ашигласан датасетүүдийг тэмдэглэх
        now = datetime.now(timezone.utc)
        async with AsyncSessionLocal() as db:
            await db.execute(
                update(TrainingSample)
                .where(TrainingSample.used_in_training.is_(False))
                .values(used_in_training=True, trained_at=now)
            )
            await db.commit()

        # 4. Donut Singleton-ийг цэвэрлэх → дараагийн хүсэлтэд шинэ загвар ачаалагдана
        import app.ml.donut_model as _donut_mod
        _donut_mod._predictor_instance = None
        logger.info(
            "Шинэ Donut загвар %s-д хадгалагдлаа. Дараагийн хүсэлтэд ачаалагдана.",
            settings.DONUT_OUTPUT_MODEL_DIR,
        )

    except Exception as exc:
        logger.error("Дахин сургалтад алдаа гарлаа: %s", exc)


# ─── FastAPI lifespan-д холбох функцууд ──────────────────────────────────────

def start_scheduler() -> None:
    """FastAPI lifespan startup дотор дуудна."""
    from app.core.config import settings

    scheduler = get_scheduler()
    scheduler.add_job(
        _check_and_retrain,
        trigger="cron",
        hour=settings.RETRAIN_CRON_HOUR,
        minute=0,
        id="auto_retrain",
        replace_existing=True,
        misfire_grace_time=3600,  # 1 цагийн доторх "алдагдсан" ажлыг гүйцэтгэнэ
    )
    scheduler.start()
    logger.info(
        "Хуваарьт сургалт идэвхжлээ (шөнийн %02d:00 цагт, босго=%d датасет).",
        settings.RETRAIN_CRON_HOUR,
        settings.AUTO_RETRAIN_THRESHOLD,
    )


def stop_scheduler() -> None:
    """FastAPI lifespan shutdown дотор дуудна."""
    scheduler = get_scheduler()
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Хуваарьт сургалт зогсоогдлоо.")
