"""
Клиент Lumean Public API (TTS через шаблоны + заказы)
Документация: https://api.lumean.app/docs/llm.md
"""

import time
import requests
from typing import Optional, List, Dict, Any
from pathlib import Path

BASE = "https://api.lumean.app/api/public"


class LumeanError(Exception):
    def __init__(self, message: str, status: int = 0, data: Any = None):
        super().__init__(message)
        self.status = status
        self.data = data


class LumeanClient:
    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        url = f"{BASE}{path}"
        r = self.session.request(method, url, timeout=60, **kwargs)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}

        if r.status_code >= 400:
            msg = data.get("message") if isinstance(data, dict) else str(data)
            raise LumeanError(msg or f"HTTP {r.status_code}", status=r.status_code, data=data)
        return data

    # ── Templates ──────────────────────────────────────────────

    def list_templates(self, page: int = 1, per_page: int = 50) -> Dict[str, Any]:
        """Список шаблонов пользователя."""
        return self._request("GET", f"/templates?page={page}&per_page={per_page}")

    def get_template(self, template_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/templates/{template_id}")

    def create_template(
        self,
        name: str,
        voice_id: str,
        service_key: str = "elevenlabs",
        model_id: str = "eleven_multilingual_v2",
        mode: str = "mode_v1",
        **extra_settings,
    ) -> Dict[str, Any]:
        """Создать TTS-шаблон с указанным голосом."""
        body = {
            "service_key": service_key,
            "name": name,
            "config": {
                "tts_settings": {
                    "mode": mode,
                    "model_id": model_id,
                    "voice_id": voice_id,
                    **extra_settings,
                }
            },
        }
        return self._request("POST", "/templates", json=body)

    # ── Voices (для удобства выбора) ───────────────────────────

    def list_elevenlabs_voices(self, page: int = 0, page_size: int = 50) -> Dict[str, Any]:
        return self._request(
            "GET",
            f"/voices/elevenlabs/library?page={page}&page_size={page_size}",
        )

    # ── Orders (TTS) ───────────────────────────────────────────

    def create_tts_order(
        self,
        template_id: str,
        text: str,
        language_code: str | None = None,
    ) -> Dict[str, Any]:
        """
        Создать заказ озвучки.
        Текст — в input_text.
        language_code (ISO-639-1: pl, tr, ar, ja…) передаём через config_override,
        чтобы TTS читал числа и текст на нужном языке, а не на языке шаблона по умолчанию.
        """
        body: Dict[str, Any] = {
            "template_id": template_id,
            "input_text": text,
        }
        if language_code:
            body["config_override"] = {
                "tts_settings": {
                    "language_code": language_code,
                }
            }
        return self._request("POST", "/orders", json=body)

    def get_order(self, order_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/orders/{order_id}")

    def wait_order(
        self,
        order_id: str,
        poll_interval: float = 2.0,
        timeout: float = 600.0,
        on_progress=None,
    ) -> Dict[str, Any]:
        """Ждать завершения заказа. Возвращает финальный объект order."""
        start = time.time()
        while True:
            data = self.get_order(order_id)
            order = data.get("data") or data
            status = order.get("status") or order.get("order", {}).get("status")

            if on_progress:
                on_progress(status, order)

            if status in ("completed", "result_delivered", "failed", "cancelled", "compensated"):
                return order

            if time.time() - start > timeout:
                raise LumeanError(f"Timeout waiting for order {order_id}", data=order)

            time.sleep(poll_interval)

    def get_download_url(self, file_path: str) -> str:
        """Получить временную ссылку на скачивание файла результата."""
        data = self._request("POST", "/storage/url", json={"path": file_path})
        # Обычно data.url или data.data.url
        if isinstance(data, dict):
            if "url" in data:
                return data["url"]
            if "data" in data and isinstance(data["data"], dict) and "url" in data["data"]:
                return data["data"]["url"]
        raise LumeanError("Не удалось получить URL скачивания", data=data)

    def download_file(self, file_path: str, dest: Path) -> Path:
        url = self.get_download_url(file_path)
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest

    def synthesize(
        self,
        template_id: str,
        text: str,
        dest: Path,
        on_progress=None,
        language_code: str | None = None,
    ) -> Path:
        """
        Полный цикл: создать заказ → дождаться → скачать mp3/wav.
        """
        create_resp = self.create_tts_order(template_id, text, language_code=language_code)
        order = create_resp.get("data") or create_resp
        order_id = order.get("id") or order.get("order", {}).get("id")
        if not order_id:
            raise LumeanError("Не получен order_id", data=create_resp)

        final = self.wait_order(order_id, on_progress=on_progress)
        status = final.get("status")
        if status not in ("completed", "result_delivered"):
            raise LumeanError(f"Заказ завершился со статусом {status}", data=final)

        # Ищем файл результата
        result = final.get("result") or {}
        files = result.get("files") or []
        if not files:
            # Иногда файлы лежат прямо в items
            items = final.get("items") or []
            for item in items:
                rf = item.get("result_file")
                if rf:
                    files.append(rf)

        if not files:
            raise LumeanError("В результате заказа нет файлов", data=final)

        # Берём первый аудиофайл
        audio_path = files[0]
        return self.download_file(audio_path, dest)
