import sys
import os
import glob
import whisper
import openai

# === CONFIG ===
# Change to your target directory, or leave as "." for current folder
target_dir = "."

if len(sys.argv) > 1:
    target_dir = sys.argv[1]

mov_files = glob.glob(os.path.join(target_dir, "*.mov"))

if not mov_files:
    print("No .mov files found.")
    sys.exit(0)

# === Batch Processing ===
for input_mov in mov_files:
    basename = os.path.basename(input_mov)
    base, ext = os.path.splitext(basename)
    if base.endswith('_480p'):
        base = base[:-6]  # remove '_480p'
    output_prefix = base
    basepath = os.path.dirname(input_mov)
    good_transcript_path = os.path.join(basepath, f"{output_prefix}.WM.transcript.txt")
    summary_path = os.path.join(basepath, f"{output_prefix}.gpt4o.summary.txt")

    # Check if both outputs exist
    if os.path.exists(good_transcript_path) and os.path.exists(summary_path):
        print(f"Skipping {basename}: both outputs already exist.")
        continue

    print(f"Processing {basename}...")

    # 1. Transcribe with Whisper (auto language detection)
    model = whisper.load_model("medium")
    result = model.transcribe(input_mov)
    transcript = result["text"]
    detected_lang = 'zh' # or: result.get("language", "en")
    print(f"Detected language for {basename}: {detected_lang}")

    # 2. Prompts by language
    if detected_lang.startswith('zh'):
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
    elif detected_lang.startswith('en'):
        clean_prompt = f"""Please lightly edit the following transcript: remove fillers and add correct punctuation, but keep the content as close to the original as possible.
---
{transcript}
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
        long_summary_prompt = f"""Summarize the main points of the following transcript in less than 500 words (bullet points preferred):
---
{{content}}
"""

    # 3. Clean transcript (punctuation etc.)
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

    # 7. Save files
    with open(good_transcript_path, 'w', encoding='utf-8') as f:
        f.write(good_transcript)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(long_summary)

    print(f"Saved cleaned transcript: {good_transcript_path}")
    print(f"Saved summary: {summary_path}")

print("Batch processing complete.")
