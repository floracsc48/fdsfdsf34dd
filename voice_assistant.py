"""Голосовой ассистент с активационными словами.

Скрипт слушает микрофон, распознаёт речь через Vosk и реагирует только
на заранее определённые активационные слова ("гостлик", "горилка",
"горилла" и их созвучные варианты). После активации запись продолжается
до тех пор, пока в эфире не наступит тишина длительностью 5 секунд.

Перед запуском:
1. Установите зависимости::

       python -m venv .venv
       source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
       pip install -r requirements.txt

2. Скачайте и распакуйте русскую модель Vosk, например
   `vosk-model-small-ru-0.22`, и укажите путь к каталогу модели через
   переменную окружения ``VOSK_MODEL_PATH`` или поместите модель в папку
   ``models/vosk-model-small-ru-0.22``.

3. Для проверки звуковых сигналов убедитесь, что устройство вывода звука
   доступно (скрипт воспроизводит короткий звуковой сигнал при активации
   и завершении записи).

Скрипт ориентирован на настольный запуск, но архитектурно его можно
адаптировать для микроконтроллера с поддержкой Python (например ESP32 с
Micropython + внешнее распознавание) — основная логика вынесена в
читаемый код.
"""
from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import simpleaudio as sa
from colorama import Fore, Style, init
from vosk import KaldiRecognizer, Model

# --- Настройки ---
SAMPLE_RATE = 16_000
CHANNELS = 1
BLOCK_SIZE = 8_000
SILENCE_TIMEOUT = 5.0  # ожидание тишины после последнего слова
POST_PHRASE_TIMEOUT = 3.0  # пауза для продолжения фразы

ACTIVATION_WORDS: Tuple[str, ...] = (
    "гостлик",
    "гостли",
    "гостлика",
    "гостликс",
    "гостлэк",
    "гостляк",
    "гостлёк",
    "гостелика",
    "гостылик",
    "гаслик",
    "гаслика",
    "гаслэк",
    "гослик",
    "гослика",
    "гослык",
    "гослыка",
    "гослико",
    "госликос",
    "госелика",
    "гостлка",
    "гостилка",
    "гошлика",
    "гошлык",
    "гошелка",
    "горилка",
    "горилкау",
    "горилкас",
    "горылка",
    "горылкау",
    "горылкас",
    "горелка",
    "горелко",
    "горелкас",
    "горилла",
    "гориллао",
    "гориллас",
    "горила",
    "гориля",
    "горилька",
    "горилки",
    "горелочка",
    "горилочка",
    "горилко",
    "горилько",
    "гарилка",
    "гарилла",
    "гарелка",
    "гарилочка",
    "корилка",
    "корилла",
)

IGNORED_EXAMPLES: Tuple[str, ...] = (
    "алиса",
    "окей гугл",
    "маруся",
    "салют",
    "джарвис",
    "привет",
    "эй",
)

START_BEEP = {"frequency": 1400, "duration": 0.12}
END_BEEP = {"frequency": 700, "duration": 0.18}

# --- Вспомогательные функции ---

def load_model() -> Model:
    """Загружает модель Vosk, используя переменную окружения или путь по умолчанию."""
    env_path = os.getenv("VOSK_MODEL_PATH")
    candidate_paths = []
    if env_path:
        candidate_paths.append(Path(env_path))
    candidate_paths.extend(
        Path(p)
        for p in (
            "models/vosk-model-small-ru-0.22",
            "models/vosk-model-ru-0.42",
        )
    )

    for path in candidate_paths:
        if path.exists():
            return Model(str(path))

    raise RuntimeError(
        "Не найдена модель Vosk. Укажите путь через VOSK_MODEL_PATH или "
        "распакуйте модель в ./models/vosk-model-small-ru-0.22"
    )


def play_beep(*, frequency: int, duration: float, volume: float = 0.4) -> None:
    """Воспроизводит короткий сигнал для обратной связи пользователю."""
    sample_rate = 44_100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    waveform = np.sin(2 * np.pi * frequency * t) * volume
    audio = np.int16(waveform * 32767)
    try:
        sa.play_buffer(audio, 1, 2, sample_rate)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Не удалось воспроизвести звук: {exc}", file=sys.stderr)


def print_activation_word_status() -> None:
    """Печатает подсказку с активными и игнорируемыми словами."""
    init(autoreset=True)
    green_words = ", ".join(f"{Fore.GREEN}{w}{Style.RESET_ALL}" for w in ACTIVATION_WORDS)
    red_words = ", ".join(f"{Fore.RED}{w}{Style.RESET_ALL}" for w in IGNORED_EXAMPLES)
    print(f"Слушаю только: {green_words}")
    print(f"Игнорирую: {red_words}\n")


def normalize_words(text: str) -> List[str]:
    return re.findall(r"[а-яa-zё]+", text.lower())


def detect_activation(words: Iterable[str]) -> Optional[str]:
    for word in words:
        if word in ACTIVATION_WORDS:
            return word
    return None


def strip_before_activation(text: str, activation: str) -> str:
    parts = text.split(activation, 1)
    if len(parts) == 2:
        return activation + parts[1]
    return text


def remove_activation(text: str, activation: str) -> str:
    parts = text.split(activation, 1)
    if len(parts) == 2:
        return parts[1].strip()
    return text


class RecorderState:
    """Хранит состояние текущей сессии распознавания."""

    def __init__(self) -> None:
        self.active = False
        self.transcript_chunks: List[str] = []
        self.current_partial = ""
        self.last_speech_ts = 0.0
        self.post_phrase_deadline: Optional[float] = None
        self.activation_word: Optional[str] = None
        self.print_lock = threading.Lock()

    def start(self, activation_word: str) -> None:
        with self.print_lock:
            self.active = True
            self.transcript_chunks = [activation_word]
            self.current_partial = ""
            self.activation_word = activation_word
            self.last_speech_ts = time.monotonic()
            self.post_phrase_deadline = None
            play_beep(**START_BEEP)
            print(
                f"{Fore.GREEN}Активация по слову: {activation_word}{Style.RESET_ALL}"
            )
            sys.stdout.write(f"{activation_word} ")
            sys.stdout.flush()

    def register_partial(self, text: str) -> None:
        if not self.active:
            return
        text = text.strip()
        if not text:
            return
        self.last_speech_ts = time.monotonic()
        addition = text
        if text.startswith(self.current_partial):
            addition = text[len(self.current_partial) :]
        elif self.current_partial.startswith(text):
            addition = ""
        with self.print_lock:
            if addition:
                sys.stdout.write(addition)
                sys.stdout.flush()
        self.current_partial = text

    def register_final(self, text: str) -> None:
        if not self.active:
            return
        text = text.strip()
        if not text:
            return
        self.last_speech_ts = time.monotonic()
        with self.print_lock:
            if self.current_partial.strip() != text:
                prefix = "" if not self.transcript_chunks else " "
                sys.stdout.write(prefix + text)
                sys.stdout.flush()
            self.transcript_chunks.append(text)
        self.current_partial = ""
        self.post_phrase_deadline = time.monotonic() + POST_PHRASE_TIMEOUT

    def finish_if_timeout(self) -> Optional[str]:
        if not self.active:
            return None
        now = time.monotonic()
        if now - self.last_speech_ts < SILENCE_TIMEOUT:
            return None
        if self.post_phrase_deadline and now < self.post_phrase_deadline:
            return None
        if self.current_partial:
            self.transcript_chunks.append(self.current_partial.strip())
        message = " ".join(chunk.strip() for chunk in self.transcript_chunks if chunk.strip())
        with self.print_lock:
            print()
            if message:
                print(f"{Fore.GREEN}{message}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Тишина, данных нет{Style.RESET_ALL}")
        play_beep(**END_BEEP)
        self.active = False
        self.transcript_chunks = []
        self.current_partial = ""
        self.activation_word = None
        self.last_speech_ts = 0.0
        self.post_phrase_deadline = None
        return message


class SpeechListener:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        self.recognizer.SetWords(True)
        self.queue: "queue.Queue[bytes]" = queue.Queue()
        self.state = RecorderState()

    def audio_callback(self, indata, frames, time_info, status):  # type: ignore[override]
        if status:
            print(status, file=sys.stderr)
        self.queue.put(bytes(indata))

    def process(self) -> None:
        print_activation_word_status()
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=CHANNELS,
            callback=self.audio_callback,
        ):
            print("Готов к работе. Скажи активационное слово...")
            while True:
                data = self.queue.get()
                if self.recognizer.AcceptWaveform(data):
                    self.handle_result(self.recognizer.Result(), final=True)
                else:
                    self.handle_result(self.recognizer.PartialResult(), final=False)
                self.state.finish_if_timeout()

    def handle_result(self, result_json: str, *, final: bool) -> None:
        result = json.loads(result_json)
        text = (result.get("text") if final else result.get("partial")) or ""
        text = text.strip()
        if not text:
            return
        words = normalize_words(text)
        if not self.state.active:
            activation = detect_activation(words)
            if activation:
                trimmed = strip_before_activation(text, activation)
                self.state.start(activation)
                remaining = remove_activation(trimmed, activation)
                if remaining:
                    if final:
                        self.state.register_final(remaining)
                    else:
                        self.state.register_partial(remaining)
            return

        if final:
            self.state.register_final(text)
        else:
            self.state.register_partial(text)


def main() -> None:
    try:
        model = load_model()
    except RuntimeError as err:
        print(err)
        sys.exit(1)

    listener = SpeechListener(model)
    try:
        listener.process()
    except KeyboardInterrupt:
        print("\nЗавершение работы по запросу пользователя.")


if __name__ == "__main__":
    main()
