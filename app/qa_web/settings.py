"""Environment-driven Flask configuration.

Kept separate from the app factory so settings can be swapped per environment
(dev / prod / test) without touching wiring code. Values come from environment
variables with sensible defaults, which is the twelve-factor convention.
"""

from __future__ import annotations

import os


def resolve_model_dir() -> str:
    """Resolve the fine-tuned model directory modularly.

    Priority:
      1. Explicit QA_MODEL_DIR environment variable
      2. Local fine-tuned checkpoint candidates (e.g. distilbert_squad_model)
      3. Default fallback path 'artifacts/qa-model'
    """
    env_dir = os.environ.get("QA_MODEL_DIR")
    if env_dir:
        return env_dir

    app_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(app_dir, "..", ".."))
    workspace_root = os.path.abspath(os.path.join(project_root, ".."))

    candidates = [
        os.path.join(workspace_root, "distilbert_squad_model"),
        os.path.join(workspace_root, "distilbert_squad_model", "distilbert_squad_model"),
        os.path.join(project_root, "distilbert_squad_model"),
        os.path.join(project_root, "distilbert_squad_model", "distilbert_squad_model"),
        os.path.join(project_root, "artifacts", "qa-model"),
        "distilbert_squad_model",
        "artifacts/qa-model",
    ]
    for cand in candidates:
        if os.path.isdir(cand) and (
            os.path.exists(os.path.join(cand, "config.json"))
            or os.path.exists(os.path.join(cand, "model.safetensors"))
            or os.path.exists(os.path.join(cand, "pytorch_model.bin"))
        ):
            return cand

    return "artifacts/qa-model"


class BaseConfig:
    # Where the fine-tuned checkpoint lives. If absent, the service falls back
    # to FALLBACK_MODEL so the API is never dead.
    MODEL_DIR: str = resolve_model_dir()
    FALLBACK_MODEL: str = os.environ.get("QA_FALLBACK_MODEL",
                                         "distilbert-base-cased-distilled-squad")
    BASELINE_MODEL: str = os.environ.get("QA_BASELINE_MODEL",
                                         "distilbert-base-cased-distilled-squad")

    # Request guard rails - reject obviously abusive payloads before inference.
    MAX_PASSAGES: int = int(os.environ.get("QA_MAX_PASSAGES", "20"))
    MAX_PASSAGE_CHARS: int = int(os.environ.get("QA_MAX_PASSAGE_CHARS", "50000"))
    MAX_QUESTION_CHARS: int = int(os.environ.get("QA_MAX_QUESTION_CHARS", "1000"))

    JSON_SORT_KEYS = False
    LOG_LEVEL: str = os.environ.get("QA_LOG_LEVEL", "INFO")

    # Load the model when the app boots (True) or on the first request (False).
    EAGER_LOAD: bool = os.environ.get("QA_EAGER_LOAD", "0") == "1"


class DevConfig(BaseConfig):
    DEBUG = True


class ProdConfig(BaseConfig):
    DEBUG = False


class TestConfig(BaseConfig):
    TESTING = True
    DEBUG = True
    EAGER_LOAD = False


_CONFIGS = {"dev": DevConfig, "development": DevConfig,
            "prod": ProdConfig, "production": ProdConfig,
            "test": TestConfig, "testing": TestConfig}


def resolve_config(name: str | None = None):
    """Pick a config class by name or the QA_ENV environment variable."""
    key = (name or os.environ.get("QA_ENV", "dev")).lower()
    return _CONFIGS.get(key, DevConfig)
