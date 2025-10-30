"""Configuration settings for the RAG system."""

import os
from typing import List, Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Application
    app_name: str = Field(default="Multimodal Agentic RAG System", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")

    # UI Configuration
    reflex_host: str = Field(default="0.0.0.0", env="REFLEX_HOST")
    reflex_port: int = Field(default=3000, env="REFLEX_PORT")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:password@localhost:5432/rag_system",
        env="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

    # Vector Database
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_collection_name: str = Field(
        default="multimodal_documents", env="QDRANT_COLLECTION_NAME"
    )
    qdrant_api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")

    # AI/ML API Keys
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    huggingface_api_key: Optional[str] = Field(
        default=None, env="HUGGINGFACE_API_KEY"
    )

    # Local Models
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    use_local_models: bool = Field(default=False, env="USE_LOCAL_MODELS")

    # Advanced Embedding Models (LFM2-ColBERT-350M or better)
    text_embedding_model: str = Field(
        default="BAAI/bge-m3", env="TEXT_EMBEDDING_MODEL"
    )
    colbert_model: str = Field(
        default="colbert-ir/colbertv2.0", env="COLBERT_MODEL"
    )
    image_embedding_model: str = Field(
        default="openai/clip-vit-large-patch14", env="IMAGE_EMBEDDING_MODEL"
    )
    audio_embedding_model: str = Field(
        default="openai/whisper-large-v3", env="AUDIO_EMBEDDING_MODEL"
    )

    # LLM Configuration
    default_llm_provider: Literal["openai", "anthropic", "ollama", "gpt_oss"] = Field(
        default="gpt_oss", env="DEFAULT_LLM_PROVIDER"
    )
    default_llm_model: str = Field(
        default="microsoft/DialoGPT-medium", env="DEFAULT_LLM_MODEL"
    )
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    max_tokens: int = Field(default=4096, env="MAX_TOKENS")

    # Agent Configuration
    agent_max_iterations: int = Field(default=10, env="AGENT_MAX_ITERATIONS")
    agent_timeout_seconds: int = Field(default=300, env="AGENT_TIMEOUT_SECONDS")
    enable_tool_use: bool = Field(default=True, env="ENABLE_TOOL_USE")

    # File Upload - Extended for code files
    max_file_size_mb: int = Field(default=100, env="MAX_FILE_SIZE_MB")
    allowed_file_types: List[str] = Field(
        default=[
            # Text formats
            ".txt", ".md", ".rst",
            # Documents
            ".pdf", ".docx", ".doc", ".odt",
            # Web formats
            ".html", ".htm", ".xml",
            # Code files
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
            ".cs", ".php", ".rb", ".go", ".rs", ".swift", ".kt", ".scala",
            ".sql", ".sh", ".bash", ".yml", ".yaml", ".json", ".toml",
            # Media
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
            ".wav", ".mp3", ".m4a", ".flac", ".ogg",
            ".mp4", ".avi", ".mov", ".mkv", ".webm"
        ],
        env="ALLOWED_FILE_TYPES",
    )
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")

    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")

    # Security
    secret_key: str = Field(
        default="your_secret_key_here_change_in_production", env="SECRET_KEY"
    )
    jwt_secret_key: str = Field(
        default="your_jwt_secret_key_here", env="JWT_SECRET_KEY"
    )
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")

    # Development
    reload_on_change: bool = Field(default=True, env="RELOAD_ON_CHANGE")
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS",
    )

    # Production
    workers: int = Field(default=4, env="WORKERS")
    worker_class: str = Field(
        default="uvicorn.workers.UvicornWorker", env="WORKER_CLASS"
    )

    class Config:
        """Pydantic config."""

        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        """Initialize settings."""
        super().__init__(**kwargs)
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)


# Global settings instance
settings = Settings()