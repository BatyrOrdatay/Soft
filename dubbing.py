"""
Пайплайн дубляжа:
1. Транскрипция (OpenAI Whisper)
2. Перевод + адаптация длины (GPT)
3. Озвучка сегментов через Lumean (по выбранному шаблону)
4. Time-stretch под оригинальные тайминги
5. Сборка финальной дорожки
"""

import json
import time
import uuid
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from openai import OpenAI
from pydub import AudioSegment

from config import get_config, PROJECTS_DIR, UPLOADS_DIR
from lumean import LumeanClient, LumeanError

import subprocess
import tempfile


def ffmpeg_atempo(in_path: Path, out_path: Path, speed: float) -> Path:
    """
    Изменить скорость речи с сохранением тона (ffmpeg atempo).
    speed > 1 = быстрее и короче, speed < 1 = медленнее и длиннее.
    atempo принимает только 0.5..2.0 за раз — цепочкуем фильтры.
    """
    if abs(speed - 1.0) < 0.02:
        if in_path.resolve() != out_path.resolve():
            out_path.write_bytes(in_path.read_bytes())
        return out_path

    factors = []
    s = speed
    # разбиваем на множители в диапазоне 0.5..2.0
    while s > 2.0:
        factors.append(2.0)
        s /= 2.0
    while s < 0.5:
        factors.append(0.5)
        s /= 0.5
    factors.append(max(0.5, min(2.0, s)))

    filt = ",".join(f"atempo={f:.5f}" for f in factors)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(in_path),
        "-filter:a", filt,
        "-vn", "-acodec", "libmp3lame", "-b:a", "192k",
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not out_path.exists():
        # fallback: без изменения скорости
        out_path.write_bytes(in_path.read_bytes())
    return out_path


from pipeline import (
    current_progress,
    update_progress,
    check_cancelled,
    transcribe_audio,
    get_audio_duration,
)


@dataclass
class DubSegment:
    index: int
    start: float
    end: float
    duration: float
    source_text: str
    target_text: str = ""
    audio_path: Optional[str] = None
    stretch_ratio: float = 1.0


def translate_and_adapt(
    segments: List[dict],
    target_lang: str,
    api_key: str,
    max_stretch: float = 1.12,
) -> List[DubSegment]:
    """
    Перевод с жёстким контролем длины (isochrony) для дубляжа.
    Мастер-промпт заставляет модель укорачивать формулировки,
    чтобы озвучка укладывалась в тайминг оригинала без сильного time-stretch.
    """
    client = OpenAI(api_key=api_key)

    lang_names = {
        "es": "Spanish",
        "fr": "French",
        "pl": "Polish",
        "en": "English",
        "de": "German",
        "pt": "Portuguese",
        "it": "Italian",
        "nl": "Dutch",
        "tr": "Turkish",
        "ja": "Japanese",
        "ar": "Arabic",
    }
    # Примерная скорость речи (символов в секунду) для целевого языка
    chars_per_sec = {
        "es": 14,
        "fr": 13,
        "pl": 13,
        "en": 15,
        "de": 13,
        "pt": 14,
        "it": 14,
        "nl": 14,
        "tr": 13,
        "ja": 8,   # японский читается медленнее по символам
        "ar": 12,
        "ru": 13,
    }
    lang_name = lang_names.get(target_lang, target_lang)
    cps = chars_per_sec.get(target_lang, 13)

    result: List[DubSegment] = []
    total = len(segments)

    master = f"""You are a professional DUBBING adapter (not a literary translator).

Task: rewrite the source line into natural spoken {lang_name} for voice-over.

CRITICAL — timing (isochrony):
- The spoken line MUST fit the given duration budget.
- Prefer shorter words, drop filler, compress phrasing.
- Meaning must stay correct, but wording can change freely.
- NEVER pad. NEVER add greetings, explanations, or quotes.
- If a literal translation is longer, REWRITE shorter until it fits the character budget.
- Output ONLY the final {lang_name} line. No notes, no quotes, no length comments.

NUMBERS (very important for TTS):
- NEVER leave digits (0-9) in the output.
- Write ALL numbers as words in {lang_name} (e.g. 2024 → words, 15% → words, 3.5 → words).
- Years, amounts, percentages, ordinals — always spoken words in {lang_name}, not Russian, not digits.

Polish/German/French/Turkish often run longer than Russian — you MUST compensate by condensing."""

    for i, seg in enumerate(segments):
        check_cancelled()
        update_progress(
            "translate",
            20 + (i / max(total, 1)) * 25,
            f"Перевод {i+1}/{total} ({lang_name})",
            scenes_total=total,
            scenes_done=i,
        )

        duration = max(0.2, float(seg["end"]) - float(seg["start"]))
        # жёсткий бюджет символов с запасом под естественные паузы
        max_chars = max(12, int(duration * cps * 0.92))
        src = (seg.get("text") or "").strip()
        if not src:
            result.append(
                DubSegment(
                    index=i,
                    start=round(seg["start"], 3),
                    end=round(seg["end"], 3),
                    duration=round(duration, 3),
                    source_text="",
                    target_text="",
                )
            )
            continue

        user = (
            f"Duration budget: {duration:.2f} seconds\n"
            f"Hard limit: at most {max_chars} characters (including spaces)\n"
            f"Source (ru): {src}\n\n"
            f"Write the {lang_name} dub line now."
        )

        target_text = src
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": master},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=220,
            )
            target_text = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
        except Exception:
            target_text = src

        # Второй проход, если всё ещё длинно
        if len(target_text) > max_chars + 8:
            try:
                resp2 = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": master},
                        {
                            "role": "user",
                            "content": (
                                f"This line is TOO LONG for {duration:.2f}s (limit {max_chars} chars).\n"
                                f"Compress harder. Same meaning. Max {max_chars} characters.\n\n"
                                f"{target_text}"
                            ),
                        },
                    ],
                    temperature=0.2,
                    max_tokens=180,
                )
                shorter = (resp2.choices[0].message.content or "").strip().strip('"').strip("'")
                if shorter and len(shorter) < len(target_text):
                    target_text = shorter
            except Exception:
                pass

        result.append(
            DubSegment(
                index=i,
                start=round(seg["start"], 3),
                end=round(seg["end"], 3),
                duration=round(duration, 3),
                source_text=src,
                target_text=target_text,
            )
        )

    return result


def merge_segments_for_tts(
    segments: List[DubSegment],
    max_chunk_sec: float = 28.0,
    max_chars: int = 900,
) -> List[Dict[str, Any]]:
    """
    Склеивает короткие сегменты в крупные куски (~20-30 сек).
    Меньше заказов в Lumean = намного быстрее.
    """
    chunks = []
    cur = {
        "indices": [],
        "start": None,
        "end": None,
        "texts": [],
        "duration": 0.0,
    }

    def flush():
        if not cur["indices"]:
            return
        chunks.append({
            "indices": list(cur["indices"]),
            "start": cur["start"],
            "end": cur["end"],
            "text": " ".join(cur["texts"]).strip(),
            "duration": round(cur["end"] - cur["start"], 3) if cur["start"] is not None else cur["duration"],
        })
        cur["indices"] = []
        cur["start"] = None
        cur["end"] = None
        cur["texts"] = []
        cur["duration"] = 0.0

    for seg in segments:
        text = (seg.target_text or "").strip()
        if not text:
            continue
        would_dur = (cur["end"] - cur["start"] + seg.duration) if cur["start"] is not None else seg.duration
        would_chars = len(" ".join(cur["texts"] + [text]))
        if cur["indices"] and (would_dur > max_chunk_sec or would_chars > max_chars):
            flush()
        if cur["start"] is None:
            cur["start"] = seg.start
        cur["end"] = seg.end
        cur["indices"].append(seg.index)
        cur["texts"].append(text)
        cur["duration"] = (cur["end"] - cur["start"]) if cur["start"] is not None else 0.0

    flush()
    return chunks


def synthesize_segments(
    segments: List[DubSegment],
    template_id: str,
    out_dir: Path,
    lumean: LumeanClient,
    parallel: int = 4,
    max_stretch: float = 1.12,
    language_code: str = "en",
) -> List[Dict[str, Any]]:
    """
    Озвучка крупными кусками + параллельно.
    Возвращает список chunk-результатов (не режем на микро-сегменты — из-за этого
    раньше «плыла» скорость внутри куска).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = merge_segments_for_tts(segments)
    total = len(chunks)
    if total == 0:
        return []

    update_progress(
        "tts", 45,
        f"Озвучка: {total} кусков (вместо {len(segments)} сегментов), x{parallel}",
        scenes_total=total,
        scenes_done=0,
    )

    results: List[Optional[Dict[str, Any]]] = [None] * total
    done_count = [0]

    def work(chunk_i: int, chunk: Dict[str, Any]):
        check_cancelled()
        raw = out_dir / f"chunk_{chunk_i:04d}_raw.mp3"
        fitted = out_dir / f"chunk_{chunk_i:04d}.mp3"
        lumean.synthesize(
            template_id=template_id,
            text=chunk["text"],
            dest=raw,
            language_code=language_code,
        )

        # Подгоняем ВЕСЬ кусок к длительности слота (start..end) одним коэффициентом
        target_dur = max(0.15, float(chunk["duration"]))
        try:
            audio = AudioSegment.from_file(raw)
            actual = len(audio) / 1000.0
        except Exception:
            return chunk_i, None

        if actual < 0.05:
            return chunk_i, None

        # speed > 1 = ускорить (укоротить)
        speed = actual / target_dur
        # ограничиваем, чтобы не было жуткого слоу-мо / писка
        if speed > max_stretch:
            speed = max_stretch
        elif speed < 1.0 / max_stretch:
            speed = 1.0 / max_stretch

        ffmpeg_atempo(raw, fitted, speed)

        # если после atempo всё ещё длиннее — мягко обрезаем хвост; короче — оставляем (тишина в сборке)
        try:
            a2 = AudioSegment.from_file(fitted)
            target_ms = int(target_dur * 1000)
            if len(a2) > target_ms + 80:
                a2 = a2[:target_ms]
                a2.export(fitted, format="mp3", bitrate="192k")
        except Exception:
            pass

        return chunk_i, {
            "start": chunk["start"],
            "end": chunk["end"],
            "duration": target_dur,
            "audio_path": str(fitted),
            "indices": chunk["indices"],
        }

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(work, i, ch): i
            for i, ch in enumerate(chunks)
            if ch.get("text")
        }
        for fut in as_completed(futures):
            check_cancelled()
            try:
                chunk_i, info = fut.result()
            except LumeanError as e:
                print(f"Lumean error on chunk {futures[fut]}: {e}")
                chunk_i, info = futures[fut], None
            except Exception as e:
                print(f"TTS failed for chunk {futures[fut]}: {e}")
                chunk_i, info = futures[fut], None

            if info is not None:
                results[chunk_i] = info

            done_count[0] += 1
            update_progress(
                "tts",
                45 + (done_count[0] / total) * 35,
                f"Озвучка {done_count[0]}/{total}",
                scenes_total=total,
                scenes_done=done_count[0],
            )

    return [r for r in results if r is not None]


def time_stretch_and_assemble(
    chunks: List[Dict[str, Any]],
    total_duration: float,
    out_path: Path,
    max_stretch: float = 1.12,
) -> Path:
    """
    Собирает финальную дорожку из кусков.
    Каждый кусок уже подогнан целиком — просто кладём на свою позицию.
    Между кусками остаётся естественная тишина оригинала.
    """
    update_progress("assemble", 85, "Сборка финальной дорожки...")

    final = AudioSegment.silent(duration=int(total_duration * 1000) + 500)

    for ch in sorted(chunks, key=lambda x: x["start"]):
        path = ch.get("audio_path")
        if not path or not Path(path).exists():
            continue
        try:
            audio = AudioSegment.from_file(path)
        except Exception as e:
            print(f"skip chunk audio: {e}")
            continue

        pos_ms = int(float(ch["start"]) * 1000)
        # не вылезаем за конец слота
        slot_ms = int(float(ch["duration"]) * 1000)
        if len(audio) > slot_ms + 50:
            audio = audio[:slot_ms]

        final = final.overlay(audio, position=pos_ms)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.export(out_path, format="mp3", bitrate="192k")
    return out_path


def run_dubbing_pipeline(
    video_or_audio_path: str,
    project_name: str,
    target_lang: str,
    template_id: str,
    source_lang: str = "ru",
) -> Dict[str, Any]:
    """
    Полный пайплайн дубляжа одного языка.
    """
    cfg = get_config()
    if not cfg.is_dubbing_ready():
        raise ValueError("Нужны Lumean API Key и OpenAI API Key в Настройках")

    current_progress.reset()

    project_id = str(uuid.uuid4())[:8]
    project_dir_name = f"dub_{project_name}_{target_lang}_{project_id}"
    project_dir = PROJECTS_DIR / project_dir_name
    project_dir.mkdir(parents=True, exist_ok=True)
    segs_dir = project_dir / "segments"

    current_progress.project_id = project_id
    current_progress.project_dir_name = project_dir_name

    try:
        # 1. Транскрипция
        update_progress("transcribe", 5, "Транскрипция исходного аудио...", project_id=project_id)
        transcript = transcribe_audio(video_or_audio_path, cfg.openai_api_key, language=source_lang)
        with open(project_dir / "transcript.json", "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)

        segments_raw = transcript.get("segments") or []
        duration = transcript.get("duration") or get_audio_duration(video_or_audio_path)

        if not segments_raw:
            raise ValueError("Не удалось получить сегменты из транскрипции")

        # 2. Перевод + адаптация
        update_progress("translate", 20, f"Перевод на {target_lang}...")
        dub_segs = translate_and_adapt(
            segments_raw,
            target_lang=target_lang,
            api_key=cfg.openai_api_key,
            max_stretch=cfg.max_stretch_ratio,
        )
        with open(project_dir / "dub_segments.json", "w", encoding="utf-8") as f:
            json.dump([asdict(s) for s in dub_segs], f, ensure_ascii=False, indent=2)

        # 3. Озвучка через Lumean
        update_progress("tts", 45, "Озвучка через Lumean (пакетно + параллельно)...")
        lumean = LumeanClient(cfg.lumean_api_key)
        chunks = synthesize_segments(
            dub_segs, template_id, segs_dir, lumean,
            parallel=4,
            max_stretch=cfg.max_stretch_ratio,
            language_code=target_lang,
        )

        # 4. Сборка
        update_progress("assemble", 85, "Сборка финальной дорожки...")
        final_path = project_dir / f"dubbed_{target_lang}.mp3"
        time_stretch_and_assemble(
            chunks,
            total_duration=duration,
            out_path=final_path,
            max_stretch=cfg.max_stretch_ratio,
        )

        # Копируем оригинал для удобства
        src_ext = Path(video_or_audio_path).suffix or ".mp3"
        shutil.copy2(video_or_audio_path, project_dir / f"source{src_ext}")

        status = {
            "status": "done",
            "project_name": project_name,
            "target_lang": target_lang,
            "template_id": template_id,
            "segments_count": len(dub_segs),
            "output": str(final_path),
        }
        with open(project_dir / "status.json", "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

        download_url = f"/projects/{project_dir_name}/dubbed_{target_lang}.mp3"
        update_progress(
            "done",
            100,
            f"Готово! Дорожка: {final_path.name}",
            scenes_total=len(dub_segs),
            scenes_done=len(dub_segs),
            download_url=download_url,
        )

        return {
            "project_id": project_id,
            "project_dir": str(project_dir),
            "output_file": str(final_path),
            "download_url": f"/projects/{project_dir_name}/dubbed_{target_lang}.mp3",
            "segments_count": len(dub_segs),
            "duration": duration,
        }

    except InterruptedError:
        update_progress("cancelled", current_progress.progress, "Отменено пользователем")
        return {"cancelled": True, "project_dir": str(project_dir)}
    except Exception as e:
        update_progress("error", 0, str(e), error=str(e))
        raise
