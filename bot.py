import os
import asyncio
import json
import tempfile

import cv2
import discord
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")
MAX_SECONDS = 30
MODEL = "gemini-flash-latest"

PROMPT = (
    "以下是一段影片每秒的截圖,依序為第 0 秒、第 1 秒……"
    "請根據畫面中人物的臉部表情與肢體語言,判斷每一秒的情緒。"
    "可選情緒:開心、難過、生氣、驚訝、恐懼、厭惡、平靜、疑惑。"
    "疑惑的特徵例如:皺眉、歪頭、眼神遲疑、嘴角微抿或側向一邊。"
    "若畫面中沒有人臉,emotion 填「無人臉」。"
    "只回傳 JSON,格式:"
    '{"results":[{"second":0,"emotion":"疑惑","reason":"一句話原因"}]}'
)


def extract_frames(path):
    """Return a list of JPEG bytes, one per second."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(round(fps))
    frames, count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            h, w = frame.shape[:2]
            scale = 768 / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", frame)
            frames.append(buf.tobytes())
            if len(frames) >= MAX_SECONDS:
                break
        count += 1
    cap.release()
    return frames


def analyze(frames):
    contents = [PROMPT]
    for i, img in enumerate(frames):
        contents.append(f"第 {i} 秒:")
        contents.append(types.Part.from_bytes(data=img, mime_type="image/jpeg"))

    resp = gemini.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )
    return json.loads(resp.text)["results"]


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    for att in message.attachments:
        if not att.filename.lower().endswith(VIDEO_EXTS):
            continue

        await message.channel.send("收到影片,分析中,請稍等...")

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, att.filename)
            await att.save(path)

            try:
                frames = await asyncio.to_thread(extract_frames, path)
                if not frames:
                    await message.channel.send("無法讀取影片內容。")
                    return
                results = await asyncio.to_thread(analyze, frames)
            except Exception as e:
                await message.channel.send(f"分析失敗:{e}")
                return

        lines = [
            f"第 {r['second']} 秒:{r['emotion']}({r['reason']})"
            for r in results
        ]
        text = "\n".join(lines)
        for i in range(0, len(text), 1900):
            await message.channel.send(text[i:i + 1900])


bot.run(os.getenv("DISCORD_TOKEN"))