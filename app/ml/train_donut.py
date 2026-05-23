"""
Donut моделийг Монгол номын зургийн өгөгдөл дээр fine-tune хийж сургах скрипт.
Энэхүү скрипт нь дараах хоёр үүрэгтэй:
  1. Мэдээллийн сангаас (PostgreSQL) номын мэдээлэл болон хавтасны зургийг татаж сургалтын өгөгдөл бэлдэх.
  2. Donut моделийг PyTorch + Transformers ашиглан сургах.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Transformers болон PyTorch суулгасан эсэхийг шалгах
try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        DonutProcessor,
        VisionEncoderDecoderModel,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        default_data_collator
    )
except ImportError:
    logger.warning("Сургалт хийхэд шаардлагатай 'torch' эсвэл 'transformers' сан суугаагүй байна.")


# ─── 1. ӨГӨГДЛИЙН САНГААС ДАТАСЕТ БЭЛДЭХ СЕКЦ ───────────────────────────────

async def export_db_dataset(output_dir: str = "donut_dataset"):
    """
    PostgreSQL-ээс номнуудын мэдээлэл болон хавтасны зураг (R2) татаж 
    Donut сургах формат руу хөрвүүлнэ.
    """
    import httpx
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.sql import text
    from app.core.config import settings

    out_path = Path(output_dir)
    img_path = out_path / "images"
    img_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Өгөгдлийн сантай холбогдож байна...")
    engine = create_async_engine(settings.DATABASE_URL)
    
    query = text("SELECT title, author, cover_url FROM books WHERE deleted_at IS NULL AND cover_url IS NOT NULL")
    
    records = []
    async with engine.connect() as conn:
        result = await conn.execute(query)
        rows = result.fetchall()
        
        logger.info(f"Нийт {len(rows)} номын мэдээлэл олдлоо. Зургуудыг татаж байна...")
        
        async with httpx.AsyncClient() as client:
            for idx, row in enumerate(rows):
                title, author, cover_url = row
                if not cover_url:
                    continue
                
                file_name = f"book_{idx:05d}.jpg"
                save_to = img_path / file_name
                
                try:
                    # Зургийг R2/URL-аас татаж хадгалах
                    response = await client.get(cover_url, timeout=10.0)
                    if response.status_code == 200:
                        with open(save_to, "wb") as f:
                            f.write(response.content)
                        
                        # Donut-ийн gt_parse формат руу хөрвүүлж бүртгэнэ
                        records.append({
                            "file_name": f"images/{file_name}",
                            "ground_truth": json.dumps({
                                "gt_parse": {
                                    "title": title,
                                    "author": author
                                }
                            }, ensure_ascii=False)
                        })
                        if (idx + 1) % 10 == 0:
                            logger.info(f"{idx + 1} зураг татаж хадгаллаа...")
                except Exception as e:
                    logger.error(f"Зураг татахад алдаа гарлаа ({cover_url}): {e}")

    # metadata.jsonl файл болгож хадгалах
    manifest_file = out_path / "metadata.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    logger.info(f"Сургалтын өгөгдөл бэлэн боллоо! Байршил: {out_path}")
    logger.info(f"Нийт амжилттай бэлтгэсэн зураг: {len(records)}")


# ─── 2. DONUT DATASET АНГИ ──────────────────────────────────────────────────

class DonutDataset(Dataset):
    """PyTorch-д зориулсан Donut өгөгдлийн сангийн анги."""

    def __init__(self, dataset_dir: str, processor, max_length: int = 128):
        self.dataset_dir = Path(dataset_dir)
        self.processor = processor
        self.max_length = max_length
        self.records = []
        
        manifest_file = self.dataset_dir / "metadata.jsonl"
        if not manifest_file.exists():
            raise FileNotFoundError(f"{manifest_file} олдсонгүй! Эхлээд өгөгдлөө бэлдэнэ үү.")
            
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = self.dataset_dir / record["file_name"]
        
        # Зургийг ачаалж бэлдэх
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze()  # [3, 960, 1280]

        # ground_truth нь JSON string хэлбэртэй байдаг
        gt = json.loads(record["ground_truth"])["gt_parse"]
        
        # Текстийг Donut-ийн XML токен хэлбэрт шилжүүлэх
        # <s_book><s_title>Гарчиг</s_title><s_author>Зохиогч</s_author></s_book>
        target_sequence = (
            f"<s_book>"
            f"<s_title>{gt['title']}</s_title>"
            f"<s_author>{gt.get('author') or ''}</s_author>"
            f"</s_book>"
        )

        labels = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        labels = labels.squeeze()
        # Padding токенуудыг алдагдлыг тооцохдоо (Loss) алгасахын тулд -100 болгоно
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values, "labels": labels}


# ─── 3. СУРГАЛТЫН УРСГАЛ (TRAINING LOOP) ────────────────────────────────────

def train_model(dataset_dir: str, model_path: str, output_model_dir: str, epochs: int = 5):
    """
    Transformers Seq2SeqTrainer ашиглан Donut загварыг сургах функц.
    """
    logger.info("Сургалтыг эхлүүлж байна...")
    
    # 1. Процессор болон суурь моделийг ачаалах
    processor = DonutProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    
    # Загварт шинээр тусгай токенуудыг бүртгэх
    special_tokens = ["<s_book>", "</s_book>", "<s_title>", "</s_title>", "<s_author>", "</s_author>"]
    processor.tokenizer.add_tokens(special_tokens)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # Моделийн тохиргоо
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_book>")

    # 2. Датасет бэлтгэх
    train_dataset = DonutDataset(dataset_dir, processor)
    
    # 3. Сургалтын параметрүүдийг тохируулах
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,  # GPU санах ойд тааруулж өөрчилнө
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_total_limit=2,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=torch.cuda.is_available(),  # GPU-тэй үед хурдасгагч
        dataloader_num_workers=2 if os.name != 'nt' else 0,
        report_to="none"
    )

    # 4. Trainer-ийг үүсгэж ажиллуулах
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    
    logger.info("Моделийг сургаж байна...")
    trainer.train()
    
    # 5. Амжилттай сургасан загвараа хадгалах
    logger.info(f"Сургаж дууссан моделийг {output_model_dir} хавтаст хадгалж байна...")
    model.save_pretrained(output_model_dir)
    processor.save_pretrained(output_model_dir)
    logger.info("Сургалт амжилттай дууслаа!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Donut модель сургах скрипт")
    parser.add_argument("--action", choices=["export", "train"], required=True, help="Гүйцэтгэх үйлдэл")
    parser.add_argument("--data_dir", default="donut_dataset", help="Өгөгдөл хадгалах/унших хавтас")
    parser.add_argument("--model_path", default="naver-clova-ix/donut-base-sys", help="Суурь моделийн нэр")
    parser.add_argument("--output_path", default="app/ml/models/donut-khonkheree", help="Сургасан загвар хадгалах байршил")
    parser.add_argument("--epochs", type=int, default=5, help="Сургах давталтын (epoch) тоо")
    
    args = parser.parse_args()
    
    if args.action == "export":
        import asyncio
        asyncio.run(export_db_dataset(args.data_dir))
    elif args.action == "train":
        train_model(args.data_dir, args.model_path, args.output_path, args.epochs)
