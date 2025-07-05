import sys
import os
import re
import whisper
import openai

if len(sys.argv) < 2:
    print("Usage: python this_script.py input_file (audio/video file)")
    sys.exit(1)

input_mov = sys.argv[1]
if not os.path.exists(input_mov):
    print(f"File not found: {input_mov}")
    sys.exit(1)

# 檔名處理，移除 _480p
basename = os.path.basename(input_mov)
base, ext = os.path.splitext(basename)
if base.endswith('_480p'):
    base = base[:-6]
output_prefix = base
basepath = os.path.dirname(input_mov)
good_transcript_path = os.path.join(basepath, f"{output_prefix}.WM.transcript.txt")
summary_path = os.path.join(basepath, f"{output_prefix}.gpt4o.summary.txt")

if os.path.exists(good_transcript_path) and os.path.exists(summary_path):
    print(f"Warning: Both '{good_transcript_path}' and '{summary_path}' already exist. Exiting.")
    sys.exit(0)

# 1. Whisper 辨識
model = whisper.load_model("medium")
result = model.transcribe(input_mov)
transcript = result["text"]
print("語音辨識完成。")

# 2. 提示詞（只做中文）
clean_prompt = f"""請修飾下面的逐字稿：
- 盡量保留原意
- 去除贅字
- 加上正確的標點符號
- 修正常見錯字（例如：錯別字、同音字、口誤導致的打錯字）
---
{transcript}
"""

long_summary_prompt = f"""請根據下面的與心理師談話的逐字稿，寫一段500字以內的摘要（用字自然一點，不要有開會的感覺，重點條列，修正常見錯別字、類似音的字 例如：產修、殘修 其實都是禪修），繁體中文：
---
{{content}}
"""

# 3. 修飾逐字稿
response_clean = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "你是一個優秀的中文逐字稿修飾助手。"},
        {"role": "user", "content": clean_prompt},
    ],
    temperature=0.2
)
good_transcript = response_clean.choices[0].message.content.strip()

# 4. 500字內摘要
prompt_long_summary_filled = long_summary_prompt.format(content=good_transcript)
response_long_summary = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "你是一位會議談話內容摘要助手。"},
        {"role": "user", "content": prompt_long_summary_filled},
    ],
    temperature=0.2
)
long_summary = response_long_summary.choices[0].message.content.strip()

# 5. 儲存結果
with open(good_transcript_path, 'w', encoding='utf-8') as f:
    f.write(good_transcript)
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(long_summary)

print(f"修飾逐字稿: {good_transcript_path}")
print(f"重點摘要: {summary_path}")
