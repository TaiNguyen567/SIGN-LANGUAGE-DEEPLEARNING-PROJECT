"""Local offline speech using the operating system's available voices."""

from __future__ import annotations

import threading
import time


class Speaker:
    """Lazy pyttsx3 wrapper so missing speech support does not break recognition."""

    def __init__(self, *, rate: int = 165, volume: float = 1.0, voice_id: str | None = None) -> None:
        self.rate = int(rate)
        self.volume = min(1.0, max(0.0, float(volume)))
        self.voice_id = voice_id
        self._engine = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self.error = ""

    def available_voices(self) -> list[dict[str, str]]:
        engine = self._get_engine()
        return [{"id": voice.id, "name": voice.name} for voice in engine.getProperty("voices")]

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        self.stop()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=3.0)
        self.error = ""
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._speak_worker, args=(text,), daemon=True, name="local-tts")
        self._worker.start()

    def stop(self) -> None:
        self._stop_event.set()

    def close(self, timeout: float = 3.0) -> bool:
        self.stop()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.0, timeout))
        return worker is None or not worker.is_alive()

    @property
    def speaking(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _speak_worker(self, text: str) -> None:
        engine = None
        try:
            engine = self._get_engine()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            if self.voice_id:
                engine.setProperty("voice", self.voice_id)
            engine.say(text)
            engine.startLoop(False)
            while engine.isBusy():
                if self._stop_event.is_set():
                    engine.stop()
                    break
                engine.iterate()
                time.sleep(0.01)
        except Exception as exc:
            self.error = f"Local text-to-speech failed: {exc}"
        finally:
            if engine is not None:
                try:
                    engine.endLoop()
                except RuntimeError as exc:
                    self.error = self.error or f"Could not stop the speech loop: {exc}"
            self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
            except Exception as exc:
                raise RuntimeError(f"No local text-to-speech engine is available: {exc}") from exc
        return self._engine
