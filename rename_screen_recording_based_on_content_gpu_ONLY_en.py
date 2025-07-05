import sys
import os
import re
import datetime
import subprocess
import whisper
import openai

if len(sys.argv) < 2:
    print("Usage: python this_script.py input_file (audio/video file)")
    sys.exit(1)

input_mov = sys.argv[1]
if not os.path.exists(input_mov):
    print(f"File not found: {input_mov}")
    sys.exit(1)

# 2. Transcribe with Whisper (Almost Only work for English)
from pywhispercpp.model import Model
# Load a multilingual model (e.g., 'medium', 'large-v2', or 'large-v3' for best results)
model = Model('large-v3')  # Or 'medium', 'large-v2', etc.

# Transcribe the audio file. Do NOT set the 'language' parameter, so Whisper auto-detects.
segments = model.transcribe(input_mov)
result = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])

from langdetect import detect

# After getting the full_text as above:
detected_lang = detect(result)
print(f"Detected language: {detected_lang}")


# 3. Prompts by language
if detected_lang.startswith('zh'):
    clean_prompt = f"""請修飾下面的逐字稿：
- 盡量保留原意
- 去除贅字
- 加上正確的標點符號
- 修正常見錯字（例如：錯別字、同音字、口誤導致的打錯字）

---
{transcript}
"""
    summary_prompt = f"""根據下面的逐字稿，請給我一句話摘要，適合作為檔案名稱（盡量包含會議主題、重要事件或參與者），請保持在30個字以內，不要包含任何前綴，只需主題內容：
---
{{content}}
"""
    long_summary_prompt = f"""請根據下面的的逐字稿(可能是與心理師談話)，寫一段500字以內的摘要（用字自然一點，不要有開會的感覺，重點條列，修正常見錯別字、類似音的字 例如：產修、殘修 其實都是禪修），繁體中文：
---
{{content}}
"""
elif detected_lang.startswith('en'):
    clean_prompt = f"""Please lightly edit the following transcript: remove fillers and add correct punctuation, but keep the content as close to the original as possible.
---
{transcript}
"""
    summary_prompt = f"""Based on the following transcript, generate a short phrase (max 10 words) that would be suitable as a filename (preferably including the meeting topic, key event, or participants). Do not include any prefixes—output only the topic:
---
{{content}}
"""
    long_summary_prompt = f"""Summarize the main points of the following transcript in less than 500 words (bullet points preferred):
---
{{content}}
"""
else:
    # fallback (use English prompts)
    clean_prompt = f"""Please lightly edit the following transcript: remove fillers and add correct punctuation, but keep the content as close to the original as possible.
---
{transcript}
"""
    summary_prompt = f"""Based on the following transcript, generate a short phrase (max 10 words) that would be suitable as a filename (preferably including the meeting topic, key event, or participants). Do not include any prefixes—output only the topic:
---
{{content}}
"""
    long_summary_prompt = f"""Summarize the main points of the following transcript in less than 500 words (bullet points preferred):
---
{{content}}
"""

# 4. Clean transcript (punctuation etc.)
response_clean = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are an excellent transcript cleanup assistant for both English and Chinese. Add correct punctuation and remove filler words with minimal changes."},
        {"role": "user", "content": clean_prompt},
    ],
    temperature=0.2
)
good_transcript = response_clean.choices[0].message.content.strip()

# 5. <500-char summary
prompt_long_summary_filled = long_summary_prompt.format(content=good_transcript)
response_long_summary = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an assistant that summarizes meetings."},
        {"role": "user", "content": prompt_long_summary_filled},
    ],
    temperature=0.2
)
long_summary = response_long_summary.choices[0].message.content.strip()


# 6. One-line summary for filename
prompt_summary_filled = summary_prompt.format(content=long_summary)
response_summary = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an assistant that generates concise file names from transcripts."},
        {"role": "user", "content": prompt_summary_filled},
    ],
    temperature=0.2
)
summary = response_summary.choices[0].message.content.strip()
summary = re.sub(r'[\\/*?:"<>|\n\r]', '', summary)

# 7. Build new filename
basename = os.path.basename(input_mov)
date_match = re.search(r'(\d{4}-\d{2}-\d{2})', basename)
if date_match:
    date_str = date_match.group(1)
else:
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')

basepath = os.path.dirname(input_mov)
new_base = f"{date_str}_{summary}"
new_mp4_path = os.path.join(basepath, f"{new_base}.mp4")
transcript_path = os.path.join(basepath, f"{new_base}.transcript.txt")
good_transcript_path = os.path.join(basepath, f"{new_base}.clean.txt")
summary_path = os.path.join(basepath, f"{new_base}.summary.txt")

# 8. Save files
with open(transcript_path, 'w', encoding='utf-8') as f:
    f.write(transcript)
with open(good_transcript_path, 'w', encoding='utf-8') as f:
    f.write(good_transcript)
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(long_summary)

print(f"原始逐字稿: {transcript_path}")
print(f"修飾逐字稿: {good_transcript_path}")
print(f"重點摘要: {summary_path}")

# 9. Convert MOV to MP4 (ffmpeg required)
output_mp4 = os.path.splitext(input_mov)[0] + '.mp4'
if not os.path.exists(output_mp4):
    print(f"Converting {input_mov} to {output_mp4} ...")
    cmd = [
        'ffmpeg',
        '-i', input_mov,
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'veryfast',
        output_mp4
    ]
    subprocess.run(cmd, check=True)
    print(f"Converted to: {output_mp4}")
else:
    print(f"MP4 file already exists: {output_mp4}")

# 10. Rename MP4
os.rename(output_mp4, new_mp4_path)
print(f"已將影片重新命名為: {new_mp4_path}")

# 11. Rename MOV
input_ext = os.path.splitext(input_mov)[1].lower()
new_input_path = os.path.join(basepath, f"{new_base}{input_ext}")
if os.path.exists(new_input_path):
    print(f"Warning: {new_input_path} already exists. Skipping renaming original file.")
else:
    os.rename(input_mov, new_input_path)
    print(f"已將原始檔案重新命名為: {new_input_path}")
