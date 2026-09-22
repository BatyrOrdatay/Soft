"""
Пайплайн: транскрипция → сцены → промпты → локальные картинки (SDXL)
"""

import os
import json
import time
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from openai import OpenAI
from pydub import AudioSegment

from config import get_config, PROJECTS_DIR, UPLOADS_DIR

# Ленивая загрузка тяжёлых библиотек
_pipe = None
_pipe_device = None


@dataclass
class Scene:
    index: int
    start: float
    end: float
    duration: float
    text: str
    prompt: str = ""
    image_path: Optional[str] = None
    image_url: Optional[str] = None


class PipelineProgress:
    def __init__(self):
        self.stage = "idle"
        self.progress = 0.0
        self.message = ""
        self.scenes_total = 0
        self.scenes_done = 0
        self.error = None
        self.project_id = None
        self.project_dir_name = None
        self.download_url = None
        self.ready_images = []
        self.cancel_requested = False
        self.queue = []
        self.queue_index = 0
        self.download_url = None

    def to_dict(self):
        return {
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "scenes_total": self.scenes_total,
            "scenes_done": self.scenes_done,
            "error": self.error,
            "project_id": self.project_id,
            "project_dir_name": self.project_dir_name,
            "ready_images": self.ready_images,
            "cancel_requested": self.cancel_requested,
            "queue_length": len(self.queue),
            "queue_index": self.queue_index,
            "download_url": self.download_url,
        }

    def reset(self):
        self.stage = "idle"
        self.progress = 0.0
        self.message = ""
        self.scenes_total = 0
        self.scenes_done = 0
        self.error = None
        self.project_id = None
        self.project_dir_name = None
        self.ready_images = []
        self.cancel_requested = False

    def request_cancel(self):
        self.cancel_requested = True
        if self.stage not in ("idle", "done", "error", "cancelled"):
            self.stage = "cancelled"
            self.message = "Генерация отменена пользователем"


current_progress = PipelineProgress()


def update_progress(stage: str, progress: float, message: str, **kwargs):
    if current_progress.cancel_requested and stage not in ("cancelled", "done", "error"):
        current_progress.stage = "cancelled"
        current_progress.message = "Генерация отменена пользователем"
        return
    current_progress.stage = stage
    current_progress.progress = progress
    current_progress.message = message
    for k, v in kwargs.items():
        setattr(current_progress, k, v)


def check_cancelled():
    if current_progress.cancel_requested:
        raise InterruptedError("Cancelled by user")


def get_audio_duration(path: str) -> float:
    audio = AudioSegment.from_file(path)
    return len(audio) / 1000.0


def transcribe_audio(file_path: str, api_key: str, language: str = "ru") -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    file_size = os.path.getsize(file_path)

    update_progress("transcribe", 5, "Начинаю транскрипцию...")

    if file_size > 20 * 1024 * 1024:
        return transcribe_long(file_path, api_key, language)

    with open(file_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language=language,
        )

    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    update_progress("transcribe", 25, "Транскрипция завершена")
    return data


def transcribe_long(file_path: str, api_key: str, language: str = "ru") -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    audio = AudioSegment.from_file(file_path)
    chunk_ms = 10 * 60 * 1000
    all_segments = []
    full_text = []

    total_chunks = (len(audio) + chunk_ms - 1) // chunk_ms
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        check_cancelled()
        update_progress("transcribe", 5 + (i / total_chunks) * 20, f"Чанк {i+1}/{total_chunks}")
        chunk = audio[start:start + chunk_ms]
        tmp = UPLOADS_DIR / f"tmp_chunk_{i}.mp3"
        chunk.export(tmp, format="mp3", bitrate="128k")

        with open(tmp, "rb") as f:
            res = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language=language,
            )
        data = res.model_dump() if hasattr(res, "model_dump") else dict(res)
        offset = start / 1000.0
        full_text.append(data.get("text", ""))
        for seg in data.get("segments", []):
            all_segments.append({
                "start": seg["start"] + offset,
                "end": seg["end"] + offset,
                "text": seg["text"],
            })
        tmp.unlink(missing_ok=True)

    return {
        "text": " ".join(full_text),
        "segments": all_segments,
        "duration": len(audio) / 1000.0,
    }


def create_scenes(segments: List[dict], duration: float, scene_dur: float = 5.5) -> List[Scene]:
    scenes = []
    t = 0.0
    idx = 0
    while t < duration:
        end = min(t + scene_dur, duration)
        texts = [s["text"].strip() for s in segments if s["end"] > t and s["start"] < end]
        text = " ".join(texts).strip() or "(пауза)"
        scenes.append(Scene(index=idx, start=round(t, 2), end=round(end, 2),
                            duration=round(end - t, 2), text=text))
        t = end
        idx += 1
    return scenes


def generate_prompts(scenes: List[Scene], style: str, api_key: str) -> List[Scene]:
    client = OpenAI(api_key=api_key)
    system = f"""You are an expert image prompt engineer.
Given a voiceover fragment, write a short powerful English prompt for a single 16:9 image.
Style (must follow): {style}
Rules: English only, visual scene, cinematic, no text on image, 15-40 words.
Reply with ONLY the prompt."""

    total = len(scenes)
    for i, scene in enumerate(scenes):
        check_cancelled()
        update_progress("prompts", 30 + (i / total) * 15, f"Промпт {i+1}/{total}",
                        scenes_total=total, scenes_done=i)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Voiceover ({scene.start:.1f}-{scene.end:.1f}s): {scene.text}"},
                ],
                temperature=0.7,
                max_tokens=120,
            )
            scene.prompt = resp.choices[0].message.content.strip().strip('"')
        except Exception as e:
            scene.prompt = f"{style}, visual scene: {scene.text[:80]}"
    return scenes


def _load_pipeline(cfg):
    """Загрузка SDXL пайплайна (один раз)."""
    global _pipe, _pipe_device

    if _pipe is not None:
        return _pipe

    import torch
    from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline

    update_progress("images", 40, "Загружаю модель SDXL (первый раз может занять несколько минут)...")

    model_id = cfg.local_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _pipe_device = device

    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        # SDXL-Turbo и подобные
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if device == "cuda" else None,
            use_safetensors=True,
        )
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
        )

    if device == "cuda":
        if cfg.use_cpu_offload:
            # Критично для 6 ГБ VRAM
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
    else:
        pipe = pipe.to(device)

    _pipe = pipe
    update_progress("images", 44, f"Модель загружена ({device})")
    return pipe


def generate_images(scenes: List[Scene], out_dir: Path, project_dir_name: str, skip_existing: bool = True) -> List[Scene]:
    """Локальная генерация через SDXL. skip_existing=True — пропускает уже готовые картинки (для продолжения)."""
    cfg = get_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(scenes)
    if not current_progress.ready_images:
        current_progress.ready_images = []

    # Сначала подхватим уже готовые картинки в галерею
    already_done = 0
    for scene in scenes:
        img_path = out_dir / f"scene_{scene.index:04d}.png"
        if img_path.exists():
            scene.image_path = str(img_path)
            image_url = f"/projects/{project_dir_name}/images/{img_path.name}"
            scene.image_url = image_url
            if not any(x.get("index") == scene.index for x in current_progress.ready_images):
                current_progress.ready_images.append({
                    "index": scene.index,
                    "start": scene.start,
                    "end": scene.end,
                    "url": image_url,
                    "prompt": (scene.prompt or "")[:80],
                })
            already_done += 1

    if already_done:
        update_progress("images", 45 + (already_done / max(total, 1)) * 50,
                        f"Уже готово {already_done}/{total}, продолжаю...",
                        scenes_total=total, scenes_done=already_done)

    need_pipe = any(
        not (out_dir / f"scene_{s.index:04d}.png").exists()
        for s in scenes if s.prompt
    )
    pipe = _load_pipeline(cfg) if need_pipe else None

    import torch

    for i, scene in enumerate(scenes):
        check_cancelled()
        img_path = out_dir / f"scene_{scene.index:04d}.png"

        # Пропуск уже сгенерированных
        if skip_existing and img_path.exists():
            update_progress("images", 45 + ((i + 1) / total) * 50, f"Пропуск {i+1}/{total} (уже есть)",
                            scenes_total=total, scenes_done=i + 1)
            continue

        update_progress("images", 45 + (i / total) * 50, f"Картинка {i+1}/{total}",
                        scenes_total=total, scenes_done=i)
        if not scene.prompt:
            continue
        if pipe is None:
            pipe = _load_pipeline(cfg)

        try:
            generator = None
            if cfg.seed >= 0:
                generator = torch.Generator(device="cpu").manual_seed(cfg.seed + scene.index)

            result = pipe(
                prompt=scene.prompt,
                num_inference_steps=cfg.inference_steps,
                guidance_scale=cfg.guidance_scale,
                width=cfg.image_width,
                height=cfg.image_height,
                generator=generator,
            )
            image = result.images[0]
            image.save(img_path)

            scene.image_path = str(img_path)
            image_url = f"/projects/{project_dir_name}/images/{img_path.name}"
            scene.image_url = image_url

            current_progress.ready_images.append({
                "index": scene.index,
                "start": scene.start,
                "end": scene.end,
                "url": image_url,
                "prompt": scene.prompt[:80] + ("..." if len(scene.prompt) > 80 else ""),
            })
            update_progress("images", 45 + ((i + 1) / total) * 50,
                            f"Картинка {i+1}/{total}",
                            scenes_total=total, scenes_done=i + 1)
        except InterruptedError:
            raise
        except Exception as e:
            print(f"Failed scene {scene.index}: {e}")
            # Пробуем ещё раз с упрощённым промптом
            try:
                result = pipe(
                    prompt=scene.prompt[:100],
                    num_inference_steps=max(1, cfg.inference_steps - 1),
                    guidance_scale=cfg.guidance_scale,
                    width=cfg.image_width,
                    height=cfg.image_height,
                )
                result.images[0].save(img_path)
                scene.image_path = str(img_path)
                image_url = f"/projects/{project_dir_name}/images/{img_path.name}"
                scene.image_url = image_url
                current_progress.ready_images.append({
                    "index": scene.index,
                    "start": scene.start,
                    "end": scene.end,
                    "url": image_url,
                    "prompt": scene.prompt[:80],
                })
            except Exception as e2:
                print(f"Retry also failed for scene {scene.index}: {e2}")

        # Небольшая пауза, чтобы не забить VRAM
        time.sleep(0.3)

    return scenes


def run_full_pipeline(mp3_path: str, project_name: str, style: str = None) -> Dict[str, Any]:
    cfg = get_config()
    if not cfg.is_ready():
        raise ValueError("Не задан OpenAI API ключ. Зайдите в Настройки.")

    current_progress.reset()

    project_id = str(uuid.uuid4())[:8]
    project_dir_name = f"{project_name}_{project_id}"
    project_dir = PROJECTS_DIR / project_dir_name
    project_dir.mkdir(parents=True, exist_ok=True)
    images_dir = project_dir / "images"

    current_progress.project_id = project_id
    current_progress.project_dir_name = project_dir_name

    try:
        # 1. Транскрипция
        update_progress("transcribe", 2, "Транскрипция аудио...", project_id=project_id)
        transcript = transcribe_audio(mp3_path, cfg.openai_api_key)
        with open(project_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)

        # 2. Сцены
        update_progress("scenes", 28, "Создаю сцены...")
        scenes = create_scenes(
            transcript.get("segments", []),
            transcript.get("duration", 0),
            cfg.scene_duration_sec,
        )
        update_progress("scenes", 30, f"Создано {len(scenes)} сцен", scenes_total=len(scenes))

        # 3. Промпты
        style = style or cfg.default_style
        scenes = generate_prompts(scenes, style, cfg.openai_api_key)

        # 4. Локальные картинки
        scenes = generate_images(scenes, images_dir, project_dir_name)

        # Сохраняем
        scenes_data = [asdict(s) for s in scenes]
        with open(project_dir / "scenes.json", "w", encoding="utf-8") as f:
            json.dump(scenes_data, f, ensure_ascii=False, indent=2)

        shutil.copy2(mp3_path, project_dir / "audio.mp3")

        status = {
            "status": "done",
            "project_name": project_name,
            "style": style,
            "scenes_count": len(scenes),
            "model": cfg.local_model,
        }
        with open(project_dir / "status.json", "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

        ready = sum(1 for s in scenes if s.image_path)
        update_progress("done", 100, f"Готово! Картинок: {ready}/{len(scenes)}",
                        scenes_total=len(scenes), scenes_done=ready)

        return {
            "project_id": project_id,
            "project_dir": str(project_dir),
            "scenes_count": len(scenes),
            "images_ready": ready,
            "duration": transcript.get("duration", 0),
        }

    except InterruptedError:
        try:
            if "scenes" in locals():
                scenes_data = [asdict(s) for s in scenes]
                with open(project_dir / "scenes.json", "w", encoding="utf-8") as f:
                    json.dump(scenes_data, f, ensure_ascii=False, indent=2)
            shutil.copy2(mp3_path, project_dir / "audio.mp3")
            status = {
                "status": "cancelled",
                "project_name": project_name,
                "style": style or "",
                "scenes_done": current_progress.scenes_done,
                "scenes_total": current_progress.scenes_total,
            }
            with open(project_dir / "status.json", "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        update_progress("cancelled", current_progress.progress,
                        f"Отменено. Сохранено {current_progress.scenes_done} картинок",
                        error="Cancelled")
        return {"cancelled": True, "project_dir": str(project_dir)}

    except Exception as e:
        update_progress("error", 0, str(e), error=str(e))
        raise


def resume_pipeline(project_dir_name: str) -> Dict[str, Any]:
    """Продолжить отменённый/частичный проект с того места, где остановились."""
    cfg = get_config()
    if not cfg.is_ready():
        raise ValueError("Не задан OpenAI API ключ. Зайдите в Настройки.")

    project_dir = PROJECTS_DIR / project_dir_name
    if not project_dir.exists():
        raise FileNotFoundError(f"Проект не найден: {project_dir_name}")

    images_dir = project_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    current_progress.reset()
    current_progress.project_dir_name = project_dir_name
    # project_id из имени если есть
    parts = project_dir_name.rsplit("_", 1)
    current_progress.project_id = parts[-1] if len(parts) == 2 else project_dir_name

    try:
        update_progress("images", 40, "Загрузка проекта для продолжения...", project_id=current_progress.project_id)

        scenes_file = project_dir / "scenes.json"
        transcript_file = project_dir / "transcript.json"
        status_file = project_dir / "status.json"

        style = cfg.default_style
        if status_file.exists():
            with open(status_file, encoding="utf-8") as f:
                st = json.load(f)
            style = st.get("style") or style

        scenes: List[Scene] = []

        if scenes_file.exists():
            with open(scenes_file, encoding="utf-8") as f:
                raw = json.load(f)
            for s in raw:
                scenes.append(Scene(
                    index=s.get("index", 0),
                    start=s.get("start", 0),
                    end=s.get("end", 0),
                    duration=s.get("duration", 0),
                    text=s.get("text", ""),
                    prompt=s.get("prompt", ""),
                    image_path=s.get("image_path"),
                    image_url=s.get("image_url"),
                ))
        elif transcript_file.exists():
            # Есть транскрипт, но нет scenes — пересоздаём
            with open(transcript_file, encoding="utf-8") as f:
                transcript = json.load(f)
            scenes = create_scenes(
                transcript.get("segments", []),
                transcript.get("duration", 0),
                cfg.scene_duration_sec,
            )
            scenes = generate_prompts(scenes, style, cfg.openai_api_key)
        else:
            raise ValueError("В проекте нет scenes.json и transcript.json — продолжить нельзя")

        # Дозаполняем промпты, если пустые
        need_prompts = [s for s in scenes if not s.prompt]
        if need_prompts:
            update_progress("prompts", 35, f"Генерирую недостающие промпты ({len(need_prompts)})...")
            # временно только для пустых
            filled = generate_prompts(need_prompts, style, cfg.openai_api_key)
            by_idx = {s.index: s for s in filled}
            for s in scenes:
                if s.index in by_idx:
                    s.prompt = by_idx[s.index].prompt

        total = len(scenes)
        done_before = sum(1 for s in scenes if (images_dir / f"scene_{s.index:04d}.png").exists())
        update_progress("images", 45, f"Продолжаю с {done_before}/{total}...",
                        scenes_total=total, scenes_done=done_before)

        scenes = generate_images(scenes, images_dir, project_dir_name, skip_existing=True)

        scenes_data = [asdict(s) for s in scenes]
        with open(scenes_file, "w", encoding="utf-8") as f:
            json.dump(scenes_data, f, ensure_ascii=False, indent=2)

        ready = sum(1 for s in scenes if s.image_path or (images_dir / f"scene_{s.index:04d}.png").exists())
        status = {
            "status": "done",
            "project_name": project_dir_name,
            "style": style,
            "scenes_count": len(scenes),
            "model": cfg.local_model,
            "resumed": True,
        }
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

        update_progress("done", 100, f"Готово! Картинок: {ready}/{len(scenes)}",
                        scenes_total=len(scenes), scenes_done=ready)

        return {
            "project_dir": str(project_dir),
            "scenes_count": len(scenes),
            "images_ready": ready,
            "resumed": True,
        }

    except InterruptedError:
        try:
            if "scenes" in locals():
                with open(project_dir / "scenes.json", "w", encoding="utf-8") as f:
                    json.dump([asdict(s) for s in scenes], f, ensure_ascii=False, indent=2)
            status = {
                "status": "cancelled",
                "project_name": project_dir_name,
                "scenes_done": current_progress.scenes_done,
                "scenes_total": current_progress.scenes_total,
            }
            with open(project_dir / "status.json", "w", encoding="utf-8") as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        update_progress("cancelled", current_progress.progress,
                        f"Отменено. Сохранено {current_progress.scenes_done} картинок",
                        error="Cancelled")
        return {"cancelled": True, "project_dir": str(project_dir)}

    except Exception as e:
        update_progress("error", 0, str(e), error=str(e))
        raise
