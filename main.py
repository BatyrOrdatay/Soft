import os
import json
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_config, UPLOADS_DIR, PROJECTS_DIR, get_cache_stats, clear_cache
from pipeline import run_full_pipeline, resume_pipeline, current_progress, update_progress
from dubbing import run_dubbing_pipeline
from lumean import LumeanClient, LumeanError

app = FastAPI(title="BatyrSoft", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
try:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/projects", StaticFiles(directory=str(PROJECTS_DIR)), name="projects")
except Exception as _mount_err:
    print("Static mount warning:", _mount_err)


@app.get("/health")
async def health():
    return {"ok": True, "service": "batyrsoft"}


class SettingsUpdate(BaseModel):
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    lumean_api_key: Optional[str] = None
    default_style: Optional[str] = None
    scene_duration_sec: Optional[float] = None
    local_model: Optional[str] = None
    inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    use_cpu_offload: Optional[bool] = None
    capcut_projects_path: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    default_source_lang: Optional[str] = None
    max_stretch_ratio: Optional[float] = None


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/settings")
async def get_settings():
    cfg = get_config()
    data = cfg.to_public_dict()
    data["cache"] = get_cache_stats()
    return data


@app.post("/api/settings")
async def update_settings(data: SettingsUpdate):
    cfg = get_config()
    if data.openai_api_key is not None and data.openai_api_key.strip():
        if not data.openai_api_key.startswith("********"):
            cfg.openai_api_key = data.openai_api_key.strip()
    if data.anthropic_api_key is not None and data.anthropic_api_key.strip():
        if not data.anthropic_api_key.startswith("********"):
            cfg.anthropic_api_key = data.anthropic_api_key.strip()
    if data.lumean_api_key is not None and data.lumean_api_key.strip():
        if not data.lumean_api_key.startswith("********"):
            cfg.lumean_api_key = data.lumean_api_key.strip()
    if data.default_style is not None:
        cfg.default_style = data.default_style
    if data.scene_duration_sec is not None:
        cfg.scene_duration_sec = data.scene_duration_sec
    if data.local_model is not None:
        cfg.local_model = data.local_model
    if data.inference_steps is not None:
        cfg.inference_steps = data.inference_steps
    if data.guidance_scale is not None:
        cfg.guidance_scale = data.guidance_scale
    if data.use_cpu_offload is not None:
        cfg.use_cpu_offload = data.use_cpu_offload
    if data.capcut_projects_path is not None:
        cfg.capcut_projects_path = data.capcut_projects_path.strip()
    if data.image_width is not None:
        cfg.image_width = data.image_width
    if data.image_height is not None:
        cfg.image_height = data.image_height
    if data.default_source_lang is not None:
        cfg.default_source_lang = data.default_source_lang
    if data.max_stretch_ratio is not None:
        cfg.max_stretch_ratio = data.max_stretch_ratio
    cfg.save()
    return {"ok": True, "settings": cfg.to_public_dict()}


@app.get("/api/cache")
async def api_cache_stats():
    return get_cache_stats()


@app.post("/api/cache/clear")
async def api_cache_clear():
    return clear_cache()


@app.get("/api/progress")
async def get_progress():
    return current_progress.to_dict()


@app.post("/api/cancel")
async def cancel_generation():
    current_progress.request_cancel()
    return {"ok": True, "message": "Запрос на отмену отправлен"}


@app.post("/api/generate")
async def start_generate(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_name: str = Form("MyVideo"),
    style: str = Form(""),
):
    cfg = get_config()
    if not cfg.is_ready():
        raise HTTPException(400, "Сначала укажите OpenAI API ключ в Настройках (нужен для транскрипции и промптов)")

    if current_progress.stage not in ("idle", "done", "error", "cancelled"):
        raise HTTPException(400, "Уже идёт генерация. Дождитесь или отмените.")

    ext = Path(file.filename).suffix or ".mp3"
    save_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOADS_DIR / save_name
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    def task():
        try:
            run_full_pipeline(str(save_path), project_name, style or None)
        except InterruptedError:
            pass
        except Exception as e:
            update_progress("error", 0, str(e), error=str(e))

    background_tasks.add_task(task)
    return {"ok": True, "message": "Генерация запущена (локальный SDXL)"}


@app.get("/api/projects")
async def list_projects():
    projects = []
    if PROJECTS_DIR.exists():
        for p in sorted(PROJECTS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_dir():
                images = list((p / "images").glob("*.png")) if (p / "images").exists() else []
                status = "unknown"
                status_file = p / "status.json"
                if status_file.exists():
                    try:
                        with open(status_file, encoding="utf-8") as f:
                            status = json.load(f).get("status", "unknown")
                    except Exception:
                        pass
                elif (p / "scenes.json").exists() and images:
                    status = "done"
                elif images:
                    status = "partial"
                else:
                    status = "empty"

                projects.append({
                    "id": p.name,
                    "name": p.name,
                    "images_count": len(images),
                    "has_transcript": (p / "transcript.json").exists(),
                    "has_scenes": (p / "scenes.json").exists(),
                    "status": status,
                    "created": p.stat().st_mtime,
                })
    return {"projects": projects}


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str):
    pdir = PROJECTS_DIR / project_id
    if not pdir.exists():
        matches = list(PROJECTS_DIR.glob(f"*{project_id}*"))
        if not matches:
            raise HTTPException(404, "Проект не найден")
        pdir = matches[0]

    scenes = []
    scenes_file = pdir / "scenes.json"
    if scenes_file.exists():
        with open(scenes_file, encoding="utf-8") as f:
            scenes = json.load(f)

    status = {}
    status_file = pdir / "status.json"
    if status_file.exists():
        with open(status_file, encoding="utf-8") as f:
            status = json.load(f)

    images = []
    images_dir = pdir / "images"
    if images_dir.exists():
        for img in sorted(images_dir.glob("*.png")):
            images.append(f"/projects/{pdir.name}/images/{img.name}")

    return {
        "id": pdir.name,
        "path": str(pdir),
        "scenes": scenes,
        "status": status,
        "images": images,
        "audio": f"/projects/{pdir.name}/audio.mp3" if (pdir / "audio.mp3").exists() else None,
    }


@app.post("/api/projects/{project_id}/resume")
async def resume_project(project_id: str, background_tasks: BackgroundTasks):
    pdir = PROJECTS_DIR / project_id
    if not pdir.exists():
        matches = list(PROJECTS_DIR.glob(f"*{project_id}*"))
        if not matches:
            raise HTTPException(404, "Проект не найден")
        pdir = matches[0]
        project_id = pdir.name

    if current_progress.stage not in ("idle", "done", "error", "cancelled"):
        raise HTTPException(400, "Уже идёт генерация. Дождитесь или отмените.")

    cfg = get_config()
    if not cfg.is_ready():
        raise HTTPException(400, "Сначала укажите OpenAI API ключ в Настройках")

    def task():
        try:
            resume_pipeline(project_id)
        except InterruptedError:
            pass
        except Exception as e:
            update_progress("error", 0, str(e), error=str(e))

    background_tasks.add_task(task)
    return {"ok": True, "message": "Продолжение проекта запущено", "project_id": project_id}


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    pdir = PROJECTS_DIR / project_id
    if not pdir.exists():
        raise HTTPException(404, "Проект не найден")
    shutil.rmtree(pdir)
    return {"ok": True}


# ───────────────────── DUBBING (Lumean) ─────────────────────

@app.get("/api/lumean/templates")
async def list_lumean_templates():
    """Список TTS-шаблонов пользователя из Lumean."""
    cfg = get_config()
    if not cfg.lumean_api_key:
        raise HTTPException(400, "Сначала укажите Lumean API Key в Настройках")
    try:
        client = LumeanClient(cfg.lumean_api_key)
        data = client.list_templates(per_page=100)

        # Lumean может вернуть разные форматы — обрабатываем все
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            if "data" in data:
                inner = data["data"]
                if isinstance(inner, list):
                    items = inner
                elif isinstance(inner, dict):
                    items = inner.get("items") or inner.get("templates") or []
            else:
                items = data.get("items") or data.get("templates") or []

        templates = []
        for t in items:
            if not isinstance(t, dict):
                continue
            templates.append({
                "id": t.get("id"),
                "name": t.get("name") or t.get("title") or "Без названия",
                "service_key": t.get("service_key") or (t.get("config") or {}).get("service_key"),
                "created_at": t.get("created_at"),
            })
        return {"templates": templates}
    except LumeanError as e:
        raise HTTPException(e.status or 400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Ошибка Lumean: {e}")


@app.post("/api/dub")
async def start_dubbing(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    project_name: str = Form("DubVideo"),
    target_lang: str = Form("es"),
    template_id: str = Form(...),
    source_lang: str = Form("ru"),
):
    """Запуск полного пайплайна дубляжа на один язык."""
    cfg = get_config()
    if not cfg.is_dubbing_ready():
        raise HTTPException(
            400,
            "Нужны Lumean API Key и OpenAI API Key в Настройках",
        )

    if current_progress.stage not in ("idle", "done", "error", "cancelled"):
        raise HTTPException(400, "Уже идёт генерация. Дождитесь или отмените.")

    if not template_id.strip():
        raise HTTPException(400, "Выберите шаблон голоса (template_id)")

    ext = Path(file.filename).suffix or ".mp3"
    save_name = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOADS_DIR / save_name
    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    def task():
        try:
            run_dubbing_pipeline(
                str(save_path),
                project_name=project_name,
                target_lang=target_lang,
                template_id=template_id.strip(),
                source_lang=source_lang,
            )
        except InterruptedError:
            pass
        except Exception as e:
            update_progress("error", 0, str(e), error=str(e))

    background_tasks.add_task(task)
    return {"ok": True, "message": f"Дубляж на {target_lang} запущен"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(__import__("os").environ.get("PORT", "8000")), reload=False)
