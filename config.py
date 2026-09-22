import os
import json
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

CONFIG_DIR = Path.home() / ".batyrsoft"
CONFIG_FILE = CONFIG_DIR / "config.json"
DATA_DIR = Path(__file__).parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROJECTS_DIR = DATA_DIR / "projects"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = DATA_DIR / "models"

for d in [CONFIG_DIR, DATA_DIR, UPLOADS_DIR, PROJECTS_DIR, CACHE_DIR, MODELS_DIR]:
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"mkdir warning {d}: {e}")


@dataclass
class AppConfig:
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    lumean_api_key: str = ""          # Lumean X-API-KEY
    default_style: str = "cinematic, photorealistic, high detail, dramatic lighting, 16:9"
    default_source_lang: str = "ru"
    max_stretch_ratio: float = 1.15   # max speed-up/slow-down for time-stretch
    scene_duration_sec: float = 5.5
    image_width: int = 1024
    image_height: int = 576   # 16:9
    local_model: str = "stabilityai/sdxl-turbo"  # or "ByteDance/SDXL-Lightning"
    inference_steps: int = 4   # Turbo/Lightning = 1-8 steps
    guidance_scale: float = 0.0  # Turbo often uses 0
    seed: int = -1  # -1 = random
    capcut_projects_path: str = ""
    use_cpu_offload: bool = True  # важно для 6 ГБ VRAM

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls) -> "AppConfig":
        data = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        known = asdict(cls()).keys()
        cfg = cls(**{k: data.get(k, getattr(cls(), k)) for k in known})
        # Railway / env overrides (удобно для деплоя)
        if os.environ.get("OPENAI_API_KEY"):
            cfg.openai_api_key = os.environ["OPENAI_API_KEY"].strip()
        if os.environ.get("LUMEAN_API_KEY"):
            cfg.lumean_api_key = os.environ["LUMEAN_API_KEY"].strip()
        if os.environ.get("ANTHROPIC_API_KEY"):
            cfg.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"].strip()
        return cfg

    def is_ready(self) -> bool:
        # Для генерации картинок нужен OpenAI
        return bool(self.openai_api_key)

    def is_dubbing_ready(self) -> bool:
        # Для дубляжа нужен Lumean + OpenAI (транскрипция + перевод)
        return bool(self.lumean_api_key) and bool(self.openai_api_key)

    def to_public_dict(self) -> dict:
        def mask(key: str) -> str:
            if not key:
                return ""
            if len(key) < 8:
                return "********"
            return "********" + key[-4:]

        return {
            "openai_api_key_masked": mask(self.openai_api_key),
            "anthropic_api_key_masked": mask(self.anthropic_api_key),
            "lumean_api_key_masked": mask(self.lumean_api_key),
            "has_openai": bool(self.openai_api_key),
            "has_lumean": bool(self.lumean_api_key),
            "default_style": self.default_style,
            "scene_duration_sec": self.scene_duration_sec,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "local_model": self.local_model,
            "inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
            "use_cpu_offload": self.use_cpu_offload,
            "capcut_projects_path": self.capcut_projects_path,
            "default_source_lang": self.default_source_lang,
            "max_stretch_ratio": self.max_stretch_ratio,
            "is_ready": self.is_ready(),
            "is_dubbing_ready": self.is_dubbing_ready(),
        }


def get_config() -> AppConfig:
    return AppConfig.load()


def get_cache_stats() -> dict:
    total_size = 0
    file_count = 0
    for d in [UPLOADS_DIR, CACHE_DIR]:
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    total_size += f.stat().st_size
                    file_count += 1
    return {
        "size_mb": round(total_size / (1024 * 1024), 2),
        "files": file_count,
    }


def clear_cache() -> dict:
    removed = 0
    for d in [UPLOADS_DIR, CACHE_DIR]:
        if d.exists():
            for f in list(d.iterdir()):
                try:
                    if f.is_file():
                        f.unlink()
                        removed += 1
                    elif f.is_dir():
                        shutil.rmtree(f)
                        removed += 1
                except Exception:
                    pass
    return {"ok": True, "removed": removed, **get_cache_stats()}
