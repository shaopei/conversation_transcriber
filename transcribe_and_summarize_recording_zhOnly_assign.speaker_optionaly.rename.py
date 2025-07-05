import sys
import os
import subprocess
import re
import openai
from pyannote.audio import Pipeline
from pydub import AudioSegment
from datetime import datetime
import torch
from pywhispercpp.model import Model
from dotenv import load_dotenv

# Load .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    print("Error: HF_TOKEN is not set. Please check your .env file.")
    sys.exit(1)

# Start timer
start_time = datetime.now()

def elapsed():
    delta = datetime.now() - start_time
    return f"{delta.total_seconds():.1f}s"

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] (+{elapsed()}) {msg}")

def ensure_wav_mono_16k(input_path):
    import wave
    base, ext = os.path.splitext(input_path)
    out_wav = base + "_16k_mono.wav"
    if ext.lower() == ".wav":
        with wave.open(input_path, 'rb') as wf:
            if wf.getnchannels() == 1 and wf.getframerate() == 16000:
                return input_path
    log(f"Converting {input_path} to mono 16kHz WAV...")
    subprocess.run(['ffmpeg', '-y', '-i', input_path, '-ar', '16000', '-ac', '1', out_wav], check=True)
    return out_wav

def load_diarization_pipeline(token):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    pipeline.to(device)
    return pipeline

def run_diarization_and_transcription(audio_file, pipeline, whisper_model_path="large-v3", verbose=False):

    diarization = pipeline(audio_file)
    log("Speaker diarization done")

    audio = AudioSegment.from_wav(audio_file)
    whisper_model = Model(whisper_model_path)
    transcript_lines = []

    segments_list = list(diarization.itertracks(yield_label=True))
    log(f"Found {len(segments_list)} segments to transcribe.")

    for i, (turn, _, speaker) in enumerate(segments_list):
        if verbose:
            log(f"Transcribing segment {i+1} of {len(segments_list)}...")
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        segment = audio[start_ms:end_ms]
        segment_file = f"segment_{start_ms}_{end_ms}.wav"
        segment.export(segment_file, format="wav")

        try:
            segments = whisper_model.transcribe(segment_file, language='zh')
            text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
        except Exception as e:
            log(f"Error transcribing segment {segment_file}: {e}")
            os.remove(segment_file)
            continue

        os.remove(segment_file)
        if text:
            #line = f"Speaker {speaker.split('_')[-1]}: {text}"
            line = f"Speaker {speaker.split('_')[-1]}: [{turn.start:.2f}-{turn.end:.2f}] {text}"
            transcript_lines.append(line)
            if verbose:
                print(line)

    del whisper_model
    log("Whisper model transcribe done")
    return "\n".join(transcript_lines)

def load_or_generate_transcript(input_file, raw_transcript_path, pipeline, verbose=False):

    if os.path.exists(raw_transcript_path):
        log(f"Found existing raw transcript at {raw_transcript_path}, skipping audio conversion, diarization, and transcription.")
        with open(raw_transcript_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        audio_file = ensure_wav_mono_16k(input_file)
        transcript = run_diarization_and_transcription(audio_file, pipeline, verbose=verbose)
        with open(raw_transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        if audio_file.endswith("_16k_mono.wav") and os.path.exists(audio_file):
            os.remove(audio_file)
            log(f"Deleted temporary file: {audio_file}")
        return transcript

def clean_and_summarize(transcript, good_transcript_path, summary_path):
    clean_prompt = f"""請修飾下面的逐字稿：
- 盡量保留原意
- 去除贅字
- 加上正確的標點符號
- 修正常見錯字（例如：錯別字、同音字、口誤導致的打錯字）
---
{transcript}
"""
    #long_summary_prompt = f"""請根據下面的與心理師談話的逐字稿，寫一段1000字以內的摘要（用字自然，保留原本說話者的用字與感覺，不要有開會的感覺，重點條列，修正常見錯別字、類似音的字 例如：產修、殘修 其實都是禪修），繁體中文：
    long_summary_prompt = f"""請根據下面的的逐字稿(可能是與心理師談話)，以主要speaker的內容寫一段1000字以內的摘要，講述她最近的生活狀態，請把人物名稱標注在內，用字自然，不要有開會的感覺，修正常見錯別字、類似音的字 例如：產修、殘修 其實都是禪修），繁體中文：

---
{{content}}
"""

    response_clean = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "你是一個優秀的中文逐字稿修飾助手。"},
            {"role": "user", "content": clean_prompt},
        ],
        temperature=0.2
    )
    good_transcript = response_clean.choices[0].message.content.strip()

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

    with open(good_transcript_path, 'w', encoding='utf-8') as f:
        f.write(good_transcript)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(long_summary)

    log(f"修飾逐字稿: {good_transcript_path}")
    log(f"重點摘要: {summary_path}")
    return good_transcript, long_summary

def generate_filename_summary(long_summary):
    summary_prompt = f"""根據下面的逐字稿，請給我一句話摘要，適合作為檔案名稱（盡量包含主題、重要事件或被speaker提到多次的名字），請保持在30個字以內，不要包含任何前綴，只需主題內容：
---
{long_summary}
"""
    response_summary = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that generates concise file names from transcripts."},
            {"role": "user", "content": summary_prompt},
        ],
        temperature=0.2
    )
    summary = response_summary.choices[0].message.content.strip()
    summary = re.sub(r'[\\/*?:"<>|\n\r]', '', summary)
    return summary

def write_srt(transcript_lines, srt_path):
    from datetime import timedelta

    def format_timestamp(seconds):
        td = timedelta(seconds=float(seconds))
        return f"{td.seconds//3600:02}:{(td.seconds//60)%60:02}:{td.seconds%60:02},{int(td.microseconds/1000):03}"

    entries = []
    for i, line in enumerate(transcript_lines, 1):
        match = re.match(r"Speaker (\d+): \[(\d+\.\d+)-(\d+\.\d+)\] (.+)", line)
        if not match:
            continue
        speaker, start, end, text = match.groups()
        entries.append(f"{i}\n{format_timestamp(start)} --> {format_timestamp(end)}\nSpeaker {speaker}: {text}\n")

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))
    log(f"SRT subtitles saved to: {srt_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python this_script.py input_file.mov|mp4|mp3|wav [--rename --force --verbose]")
        sys.exit(1)

    input_file = sys.argv[1]
    rename_file = '--rename' in sys.argv
    verbose = '--verbose' in sys.argv
    force = '--force' in sys.argv

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)

    base = os.path.splitext(os.path.basename(input_file))[0]
    if base.endswith('_480p'):
        base = base[:-6]
    basepath = os.path.dirname(input_file)
    output_prefix = base

    raw_transcript_path = os.path.join(basepath, f"{output_prefix}.gpu.speakers.raw_transcript.txt")
    good_transcript_path = os.path.join(basepath, f"{output_prefix}.gpu.speakers.clean_transcript.txt")
    summary_path = os.path.join(basepath, f"{output_prefix}.gpu.speakers.summary.txt")

    if os.path.exists(good_transcript_path) and os.path.exists(summary_path) and not force:
        log("Both clean transcript and summary already exist. Use --force to overwrite.")
        sys.exit(0)

    pipeline = load_diarization_pipeline(HF_TOKEN)
    transcript = load_or_generate_transcript(input_file, raw_transcript_path, pipeline, verbose)

    good_transcript, long_summary = clean_and_summarize(transcript, good_transcript_path, summary_path)
    
    # Save .srt subtitles
    transcript_lines = transcript.strip().splitlines()
    srt_path = os.path.join(basepath, f"{output_prefix}.srt")
    write_srt(transcript_lines, srt_path)


    # --- Summary-based renaming ---
    if rename_file:
        # Accept YYYY-MM-DD or YYYYMMDD
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', base)
        if date_match:
            date_str = date_match.group(1)
        else:
            date_match = re.search(r'(\d{8})', base)
            if date_match:
                # Convert 20211221 -> 2021-12-21
                d = date_match.group(1)
                date_str = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')
        summary_for_name = generate_filename_summary(long_summary)
        ext = os.path.splitext(input_file)[1]
        new_base = f"{date_str}_{summary_for_name}"


        # --- Rename main media file ---
        new_file_path = os.path.join(basepath, f"{new_base}{ext}")
        if os.path.abspath(new_file_path) != os.path.abspath(input_file):
            if os.path.exists(new_file_path):
                log(f"Target file {new_file_path} already exists. Skipping rename.")
            else:
                os.rename(input_file, new_file_path)
                log(f"原始檔案為: {input_file}")
                log(f"已將原始檔案重新命名為: {new_file_path}")

        # --- Rename transcript and summary files ---
        new_raw_transcript_path = os.path.join(basepath, f"{new_base}.gpu.speakers.raw_transcript.txt")
        new_good_transcript_path = os.path.join(basepath, f"{new_base}.gpu.speakers.clean_transcript.txt")
        new_summary_path = os.path.join(basepath, f"{new_base}.gpu.speakers.summary.txt")
        new_srt_path = os.path.join(basepath, f"{new_base}.srt")
        def safe_rename(src, dst):
            if os.path.exists(src):
                if os.path.exists(dst):
                    log(f"Target file {dst} already exists. Skipping rename for transcript/summary.")
                else:
                    os.rename(src, dst)
                    log(f"Renamed: {src} -> {dst}")
            else:
                log(f"File {src} not found, cannot rename.")

        safe_rename(raw_transcript_path, new_raw_transcript_path)
        safe_rename(good_transcript_path, new_good_transcript_path)
        safe_rename(summary_path, new_summary_path)
        safe_rename(srt_path, new_srt_path)

if __name__ == "__main__":
    main()
