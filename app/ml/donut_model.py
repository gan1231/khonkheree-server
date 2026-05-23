import re
import logging
from PIL import Image

logger = logging.getLogger(__name__)

# Single instance cache to prevent reloading the model on every request
_predictor_instance = None


def get_donut_predictor(model_path: str) -> "DonutPredictor":
    """DonutPredictor-ийн ганц хувилбарыг (Singleton) буцаана."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = DonutPredictor(model_path)
    return _predictor_instance


class DonutPredictor:
    """Donut моделийг ажиллуулах, дүрсээс мэдээлэл таних үндсэн класс."""

    def __init__(self, model_path: str):
        logger.info(f"Donut моделийг ачаалж байна: {model_path}...")
        try:
            import torch
            from transformers import DonutProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            logger.error(
                "PyTorch эсвэл Transformers сан олдсонгүй! "
                "pip install torch transformers sentencepiece protobuf ажиллуулна уу."
            )
            raise e

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Hugging Face эсвэл локал файлаас процессор болон моделийг ачаалах
        self.processor = DonutProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Donut моделийг амжилттай ачааллаа. Төхөөрөмж: {self.device}")

    def predict(self, image_path: str) -> dict:
        """
        Зургийн замаар номын нэр, зохиогч зэрэг мэдээллийг ялгаж авна.
        """
        try:
            import torch
            
            image = Image.open(image_path).convert("RGB")
            
            # Зургийг моделийн оролт (pixel values) болгож бэлдэх
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Сургасан моделийн эхлэх токен
            task_prompt = "<s_book>"
            tokenizer = self.processor.tokenizer
            
            # Хэрэв токенжир дотор тусгай токен байхгүй бол суурь моделийн дагуу ажиллуулна
            if task_prompt not in tokenizer.get_added_vocab():
                decoder_input_ids = None
            else:
                decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
                decoder_input_ids = decoder_input_ids.to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self.model.config.decoder.max_position_embeddings,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=[[tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )

            # Гаралтыг текст токен болгож хөрвүүлнэ
            sequence = tokenizer.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
            
            # Эхний task токеныг хасах
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
            
            # Токен текстийг JSON болгож хөрвүүлэх
            try:
                parsed = self.processor.token2json(sequence)
            except Exception as e:
                logger.warning(f"Токеныг JSON болгоход алдаа гарлаа: {e}")
                parsed = {"text": sequence}
            
            # Хэрэв gt_parse дотор үр дүн орсон байвал хавтгайруулна
            if "gt_parse" in parsed:
                parsed = parsed["gt_parse"]
                
            # Төслийн хүлээж авах API-ийн форматад шилжүүлж цэгцлэх
            result = {
                "title": parsed.get("title") or parsed.get("name") or parsed.get("text") or None,
                "author": parsed.get("author") or parsed.get("writer") or None,
                "publisher": parsed.get("publisher") or None,
                "language": parsed.get("language") or parsed.get("lang") or None,
                "confidence": "high" if (parsed.get("title") or parsed.get("text")) else "low"
            }
            
            # Хэрэв хоосон үр дүн гарвал итгэлцүүрийг low болгоно
            if not result["title"]:
                result["confidence"] = "low"
                
            return result
            
        except Exception as e:
            logger.error(f"Donut таамаглал хийхэд алдаа гарлаа: {e}")
            return {
                "error": str(e),
                "title": None,
                "author": None,
                "confidence": "low"
            }
