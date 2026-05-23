from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "Khonkheree API"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/khonkheree"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # JWT
    JWT_SECRET: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_EXPIRE_DAYS: int = 30

    # Cloudflare R2 (S3-compatible)
    R2_ACCOUNT_ID: Optional[str] = None
    R2_ACCESS_KEY: Optional[str] = None
    R2_SECRET_KEY: Optional[str] = None
    R2_BUCKET: str = "khonkheree-covers"

    # Apple Sign In
    APPLE_TEAM_ID: Optional[str] = None
    APPLE_CLIENT_ID: str = "app.khonkheree"
    APPLE_KEY_ID: Optional[str] = None
    APPLE_PRIVATE_KEY: Optional[str] = None

    # Ollama (ML)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    VISION_MODEL: str = "qwen2.5vl:7b"

    # Donut (ML)
    USE_DONUT: bool = True
    DONUT_MODEL_PATH: str = "naver-clova-ix/donut-base-sys"

    # CORS
    ALLOWED_ORIGINS: list[str] = ["https://khonkheree.app", "http://localhost:3000"]

    class Config:
        env_file = ".env"


settings = Settings()
