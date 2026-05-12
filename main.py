import base64
import os
import re
import time
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty, Full

import cv2
import numpy as np
import pygame
import requests
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_URL = os.getenv(
    "OPENAI_URL", "https://ai.hackclub.com/proxy/v1/chat/completions"
)
# Default from live GET https://ai.hackclub.com/proxy/v1/models: strongest image+text in Gemini 3.1 line.
# Set LLM_MODEL to e.g. google/gemini-3-flash-preview for faster/cheaper narration.
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-3.1-pro-preview")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
CHARACTER_NAMES = [
    n.strip()
    for n in os.getenv("CHARACTER_NAMES", "shrit, raj, punit, shreyas").split(",")
    if n.strip()
]
BG_MUSIC_PATH = os.getenv("BG_MUSIC_PATH", "bg.mp3")

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """
You are a witty comedy narrator — think David Attenborough meets a stand-up comic. Tell a SHORT, FUNNY story about the people on camera. Epic-but-ridiculous opening sequence, like the trailer to a low-budget action movie that takes itself way too seriously.

THE CAST (the user will tell you who is visible each turn):
Possible named characters: shrit, raj, punit, shreyas. There may also be extra unnamed people in the frame. NEVER invent a name for an unnamed person — call them something funny like "the mystery guest", "the silent legend", or "this random king" for laughs.

STRUCTURE:
- On the FIRST turn you'll be asked to "open the story". Introduce each visible named character in ONE punchy line (≈8-15 words each), giving them a silly motivation, fake superpower, or absurd backstory. Then add 1-2 sentences setting up the scene comedically. Land at least one joke.
- On LATER turns, do NOT re-introduce people. Just narrate the next funny beat of the ongoing story (1-3 sentences), call back to earlier jokes, and react to what changed in the scene.

TONE:
- Playful, observational, a little dramatic, slightly self-aware. Roast the situation, never the people. PG-13 max — no slurs, no mean-spirited body stuff.
- Use words a 4th grader can understand. Short sentences hit harder.

OUTPUT RULES (very important):
- Reply with ONLY the spoken narration. Plain prose. Nothing else.
- Do NOT include stage directions, sound cues, music notes, or scene descriptions in parentheses or brackets. e.g. never write "(music swells)", "[camera pans]", "*dramatic pause*".
- Do NOT include speaker labels like "Narrator:" or "NARRATOR -".
- Do NOT use markdown: no **bold**, no *italics*, no underscores, no headings, no bullets, no quotes around the line.
- Output a single continuous paragraph, ready to be spoken aloud verbatim.
- Hard cap: 90 words. Punchy beats sleepy.
""".strip()


def start_bg_music(track: str) -> None:
    if not Path(track).exists():
        print(f"Background music file not found: {track}")
        return
    pygame.mixer.init()
    try:
        pygame.mixer.music.load(track)
        pygame.mixer.music.set_volume(0.3)
        pygame.mixer.music.play(-1)
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(10)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pygame.mixer.music.stop()
        except pygame.error:
            pass


def music_process() -> None:
    start_bg_music(BG_MUSIC_PATH)


def play_audio(client: ElevenLabs, text_input: str, audio_file_cnt: int) -> int:
    if not text_input:
        return audio_file_cnt
    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=ELEVENLABS_MODEL_ID,
            optimize_streaming_latency=0,
            output_format="mp3_22050_32",
            text=text_input,
            voice_settings=VoiceSettings(
                stability=0.55,
                similarity_boost=0.75,
                style=0.35,
                use_speaker_boost=True,
            ),
        )
        output_file = AUDIO_DIR / f"output_audio{audio_file_cnt}.mp3"
        with open(output_file, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(output_file))
        pygame.mixer.music.play()
        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(10)
        return audio_file_cnt + 1
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")
        return audio_file_cnt


_PAREN_RE = re.compile(r"\([^)]*\)")
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_CURLY_RE = re.compile(r"\{[^}]*\}")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_STAR_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_ITALIC_UNDER_RE = re.compile(r"(?<!_)_([^_\n]+)_(?!_)")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s*", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_SPEAKER_RE = re.compile(
    r"\b(?:narrator|voice[- ]?over|v\.?o\.?|scene|cue|sfx|fx)\s*[:\-–—]\s*",
    re.IGNORECASE,
)
_BACKTICK_RE = re.compile(r"`+")
_QUOTE_WRAP_RE = re.compile(r'^["“”\']\s*(.*?)\s*["“”\']$', re.DOTALL)
_MULTI_WS_RE = re.compile(r"\s+")


def clean_narration(text: str) -> str:
    """Strip stage directions, speaker labels, and markdown so only spoken narration remains."""
    if not text:
        return ""
    text = _PAREN_RE.sub("", text)
    text = _BRACKET_RE.sub("", text)
    text = _CURLY_RE.sub("", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_STAR_RE.sub(r"\1", text)
    text = _ITALIC_UNDER_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _SPEAKER_RE.sub("", text)
    text = _BACKTICK_RE.sub("", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    m = _QUOTE_WRAP_RE.match(text)
    if m:
        text = m.group(1).strip()
    return text


def enhance_image(image: np.ndarray) -> np.ndarray:
    img = np.float32(image) / 255.0
    img = cv2.pow(img, 1.5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] *= 1.15
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    enhanced = np.clip(enhanced, 0, 1)
    return (255 * enhanced).astype(np.uint8)


def resize_image(image: np.ndarray, max_width: int = 500) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    ratio = max_width / float(w)
    return cv2.resize(image, (max_width, int(h * ratio)), interpolation=cv2.INTER_AREA)


def build_user_message(base64_image: str, first_turn: bool) -> dict:
    cast = ", ".join(CHARACTER_NAMES) if CHARACTER_NAMES else "(no known cast)"
    if first_turn:
        instruction = (
            "OPEN THE STORY. Look at this frame. "
            f"The possible named cast (in any combination): {cast}. "
            "Up to 5 people may be on camera; some may not match the cast — "
            "for those, do NOT invent a name, use a funny placeholder like "
            "'the mystery guest' or 'the silent legend'. "
            "Briefly introduce each visible named character in one comedic punchy line, "
            "then set up the scene with a joke. Keep it under 90 words, plain prose, no markdown."
        )
    else:
        instruction = (
            "CONTINUE the funny story. Same cast as before. Do NOT re-introduce anyone. "
            "Narrate the next 1-3 sentence comedic beat based on what's happening in this new frame. "
            "Call back to earlier bits if you can. Under 60 words, plain prose, no markdown."
        )
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ],
    }


def call_llm(base64_image: str, script: list) -> str | None:
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY is not set; skipping LLM call.")
        return None
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    first_turn = len(script) == 0
    payload = {
        "model": LLM_MODEL,
        "messages": (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + script
            + [build_user_message(base64_image, first_turn=first_turn)]
        ),
        "max_tokens": 350,
    }
    try:
        response = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        print("Error: unexpected response:", data)
        return None
    except requests.exceptions.RequestException as e:
        print("Request error:", e)
        return None


def add_subtitle(image: np.ndarray, text: str = "", max_line_length: int = 40) -> np.ndarray:
    if not text:
        return image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    line_type = 2
    margin = 10
    line_spacing = 30
    shadow_offset = 2

    words = text.split()
    lines: list[str] = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_line_length:
            current_line += word + " "
        else:
            if current_line:
                lines.append(current_line.rstrip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.rstrip())

    start_y = image.shape[0] - line_spacing * len(lines) - margin
    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, line_type)[0]
        text_x = (image.shape[1] - text_size[0]) // 2
        text_y = start_y + i * line_spacing
        cv2.putText(
            image,
            line,
            (text_x + shadow_offset, text_y + shadow_offset),
            font,
            font_scale,
            (0, 0, 0),
            line_type,
        )
        cv2.putText(image, line, (text_x, text_y), font, font_scale, (255, 255, 255), line_type)
    return image


def open_camera() -> cv2.VideoCapture:
    backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
    return cv2.VideoCapture(0, backend)


def drain_latest(q: Queue):
    item = None
    try:
        while True:
            item = q.get_nowait()
    except Empty:
        return item


def webcam_capture(frame_queue: Queue, subtitle_queue: Queue, capture_interval: float = 5.0) -> None:
    cap = open_camera()
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    subtitle_text = "---"
    last_send = 0.0
    try:
        cv2.namedWindow("Sayonara Hokage", cv2.WINDOW_AUTOSIZE)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            now = time.time()
            if now - last_send >= capture_interval:
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except Empty:
                        break
                try:
                    frame_queue.put_nowait(frame.copy())
                    last_send = now
                except Full:
                    pass

            latest_sub = drain_latest(subtitle_queue)
            if latest_sub:
                subtitle_text = latest_sub

            display = enhance_image(frame)
            display = add_subtitle(display, subtitle_text)
            cv2.imshow("Sayonara Hokage", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def process_frames(frame_queue: Queue, subtitle_queue: Queue) -> None:
    if not ELEVENLABS_API_KEY:
        print("ELEVENLABS_API_KEY is not set; narration will be skipped.")
        client = None
    else:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    script: list = []
    audio_file_cnt = 0
    try:
        while True:
            try:
                frame = frame_queue.get(timeout=5)
            except Empty:
                continue

            print("----capturing----")
            cv2.imwrite("frame.jpg", frame)

            resized = resize_image(frame)
            ok, buffer = cv2.imencode(".jpg", resized)
            if not ok:
                continue
            base64_image = base64.b64encode(buffer).decode("utf-8")

            output = call_llm(base64_image, script)
            if output is None:
                time.sleep(2)
                continue
            script.append({"role": "assistant", "content": output})

            narration = clean_narration(output)
            print("raw:", output)
            print("narration:", narration)
            if not narration:
                continue

            subtitle_queue.put(narration)
            if client is not None:
                audio_file_cnt = play_audio(client, narration, audio_file_cnt)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error during capturing image: {e}")


def main() -> None:
    if not ELEVENLABS_API_KEY or not OPENAI_API_KEY:
        print(
            "Warning: ELEVENLABS_API_KEY and/or OPENAI_API_KEY are not set. "
            "Create a .env file (see .env.example) or export them in your shell."
        )

    frame_queue: Queue = Queue(maxsize=1)
    subtitle_queue: Queue = Queue()

    webcam_proc = Process(target=webcam_capture, args=(frame_queue, subtitle_queue))
    music_proc = Process(target=music_process)
    frames_proc = Process(target=process_frames, args=(frame_queue, subtitle_queue))

    webcam_proc.start()
    frames_proc.start()
    music_proc.start()

    try:
        webcam_proc.join()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (frames_proc, music_proc, webcam_proc):
            if proc.is_alive():
                proc.terminate()
        for proc in (frames_proc, music_proc, webcam_proc):
            proc.join(timeout=5)


if __name__ == "__main__":
    main()
