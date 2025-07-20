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
    
    # Check if the expected WAV file already exists
    if os.path.exists(out_wav):
        log(f"Found existing WAV file at {out_wav}, using it.")
        return out_wav, False  # False means we didn't create it
    
    # Check if input is already the right format
    if ext.lower() == ".wav":
        try:
            with wave.open(input_path, 'rb') as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == 16000:
                    return input_path, False  # False means we didn't create it
        except Exception as e:
            log(f"Warning: Could not check WAV format: {e}")
    
    # Convert to required format
    log(f"Converting {input_path} to mono 16kHz WAV...")
    try:
        subprocess.run(['ffmpeg', '-y', '-i', input_path, '-ar', '16000', '-ac', '1', out_wav], 
                      check=True, capture_output=True, text=True)
        return out_wav, True  # True means we created it
    except subprocess.CalledProcessError as e:
        log(f"FFmpeg conversion failed: {e}")
        log(f"FFmpeg stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        log("Error: FFmpeg not found. Please install FFmpeg.")
        raise

def load_diarization_pipeline(token):
    # Enhanced device detection for all GPU types
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon/Intel Mac with Metal
        print(f"Using device: {device} (Apple Metal GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
        print(f"Using device: {device} (NVIDIA GPU)")
    else:
        device = torch.device("cpu")  # CPU fallback
        print(f"Using device: {device} (CPU)")
    
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    pipeline.to(device)
    return pipeline

def run_diarization_and_transcription(audio_file, pipeline, whisper_model_path="large-v3", verbose=False, language=None):
    diarization = pipeline(audio_file)
    log("Speaker diarization done")

    audio = AudioSegment.from_wav(audio_file)
    whisper_model = Model(whisper_model_path)
    transcript_lines = []
    segments_list = list(diarization.itertracks(yield_label=True))
    log(f"Found {len(segments_list)} segments to transcribe.")

    # Initialize language detection variable
    detected_lang = None

    for i, (turn, _, speaker) in enumerate(segments_list):
        if verbose:
            log(f"Transcribing segment {i+1} of {len(segments_list)}...")
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        segment = audio[start_ms:end_ms]
        
        # Use safer temporary file handling to avoid I/O errors
        segment_file = None
        text = ""
        
        try:
            import tempfile
            
            # Create temporary file with proper cleanup
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                segment_file = temp_file.name
                segment.export(segment_file, format="wav")
            
            # Verify the file was created and has content
            if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                # Use the specified language (defaults to English)
                segments = whisper_model.transcribe(segment_file, language=language)
                if i == 0:
                    log(f"Using language: {language}")
                
                text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
            else:
                log(f"Warning: Temporary file {segment_file} is empty or missing")
            
        except Exception as e:
            log(f"Error transcribing segment {i+1}: {e}")
            text = ""
        finally:
            # Always clean up the temporary file
            if segment_file and os.path.exists(segment_file):
                try:
                    os.remove(segment_file)
                except OSError as cleanup_error:
                    log(f"Warning: Could not remove temp file {segment_file}: {cleanup_error}")
        
        if text:
            line = f"Speaker {speaker.split('_')[-1]}: [{turn.start:.2f}-{turn.end:.2f}] {text}"
            transcript_lines.append(line)
            if verbose:
                print(line)

    del whisper_model
    log("Whisper model transcribe done")
    return "\n".join(transcript_lines)

def load_or_generate_transcript(input_file, raw_transcript_path, pipeline, verbose=False, language=None, force=False):
    if os.path.exists(raw_transcript_path) and not force:
        log(f"Found existing raw transcript at {raw_transcript_path}, skipping audio conversion, diarization, and transcription.")
        with open(raw_transcript_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        audio_file, was_created = ensure_wav_mono_16k(input_file)
        transcript = run_diarization_and_transcription(audio_file, pipeline, verbose=verbose, language=language)
        with open(raw_transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        # Only delete the WAV file if we created it
        if was_created and audio_file.endswith("_16k_mono.wav") and os.path.exists(audio_file):
            os.remove(audio_file)
            log(f"Deleted temporary file: {audio_file}")
        elif was_created:
            log(f"Keeping WAV file: {audio_file}")
        return transcript

def clean_transcript(transcript, good_transcript_path):
    # If transcript is very long, chunk it to avoid timeout
    max_chars = 6000  # Conservative limit for transcript cleaning
    if len(transcript) > max_chars:
        log(f"Transcript is very long ({len(transcript)} chars), using chunking for cleaning")
        # Split into chunks and process each
        chunks = []
        for i in range(0, len(transcript), max_chars):
            chunk = transcript[i:i + max_chars]
            chunks.append(chunk)
        
        cleaned_chunks = []
        for i, chunk in enumerate(chunks):
            log(f"Cleaning chunk {i+1} of {len(chunks)}...")
            cleaned_chunk = clean_transcript_chunk(chunk)
            cleaned_chunks.append(cleaned_chunk)
        
        good_transcript = "\n\n".join(cleaned_chunks)
    else:
        good_transcript = clean_transcript_chunk(transcript)
    
    with open(good_transcript_path, 'w', encoding='utf-8') as f:
        f.write(good_transcript)
    log(f"Refined transcript: {good_transcript_path}")
    return good_transcript

def detect_language(text):
    """Simple language detection based on character sets"""
    if not text:
        return 'en'
    
    # Count Chinese characters (CJK Unified Ideographs)
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    total_chars = len([char for char in text if char.isalpha()])
    
    if total_chars == 0:
        return 'en'
    
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    
    if chinese_ratio > 0.3:  # If more than 30% are Chinese characters
        return 'zh'
    else:
        return 'en'

def clean_transcript_chunk(transcript_chunk):
    # Detect language and create appropriate prompt
    language = detect_language(transcript_chunk)
    log(f"Detected language for this chunk: {language}")
    
    if language.startswith('zh'):
        clean_prompt = f"""請修飾下面的逐字稿：
- 盡量保留原意
- 去除贅字
- 加上正確的標點符號
- 修正常見錯字（例如：錯別字、同音字、口誤導致的打錯字）
---
{transcript_chunk}
"""
    else:
        # English and other languages
        clean_prompt = f"""Please clean up the following transcript:
- Keep the original meaning
- Remove filler words
- Add correct punctuation
- Fix common typos and spelling errors
---
{transcript_chunk}
"""
    
    # Try gpt-4.1-mini with increasing timeouts and retries
    timeouts_to_try = [120, 180, 240]  # 2min, 3min, 4min
    max_retries = 3
    
    for attempt in range(max_retries):
        timeout_seconds = timeouts_to_try[attempt] if attempt < len(timeouts_to_try) else timeouts_to_try[-1]
        
        try:
            log(f"Attempt {attempt + 1}/{max_retries}: Trying gpt-4.1-mini with {timeout_seconds}s timeout...")
            response_clean = openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are an expert transcript editor, fluent in English, Chinese, and occasionally French. Keep the original wording as much as possible. Improve clarity and flow, and carefully check for mis-transcribed words, especially similar-sounding ones with different meanings, while preserving the speaker’s intent."},
                    {"role": "user", "content": clean_prompt},
                ],
                temperature=0.2,
                timeout=timeout_seconds
            )
            good_transcript = response_clean.choices[0].message.content
            if good_transcript is None:
                log(f"Warning: gpt-4.1-mini returned None on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    continue
                else:
                    break
            log(f"Successfully cleaned chunk using gpt-4.1-mini (attempt {attempt + 1})")
            return good_transcript.strip()
            
        except Exception as e:
            log(f"Error with gpt-4.1-mini (attempt {attempt + 1}): {e}")
            if "timeout" in str(e).lower():
                log(f"Timeout with gpt-4.1-mini on attempt {attempt + 1}, trying again with longer timeout...")
            if attempt < max_retries - 1:
                continue
            else:
                break
    
    # If all attempts fail, return original
    log("All attempts with gpt-4.1-mini failed for transcript cleaning, using original")
    return transcript_chunk

def generate_summary(good_transcript, summary_path, long_summary_prompt_file=None):
    # Detect language and create appropriate prompt
    language = detect_language(good_transcript)
    long_summary_prompt = None
    if long_summary_prompt_file:
        with open(long_summary_prompt_file, 'r', encoding='utf-8') as f:
            custom_prompt = f.read()
        if '{good_transcript}' in custom_prompt:
            long_summary_prompt = custom_prompt.replace('{good_transcript}', good_transcript)
        else:
            long_summary_prompt = custom_prompt.rstrip() + f"\n\nTranscript:\n{good_transcript}"
    else:
        if language.startswith('zh'):
            long_summary_prompt = f"""請根據下面的逐字稿，寫一段1000字以內的摘要，涵蓋所有主要speaker的內容和觀點。請：

1. 識別並總結每個主要speaker的重要觀點和貢獻
2. 描述他們討論的主題、問題或事件
3. 記錄任何重要的決定、結論或行動項目
4. 把人物名稱標注在內
5. 用字自然，不要有太生硬開會的感覺
6. 修正常見錯別字、類似音的字（例如：產修、殘修 其實都是禪修）
7. 繁體中文

---
{good_transcript}
"""
        else:
            # English and other languages
            long_summary_prompt = f"""Based on the following transcript, write a summary of up to 1000 words covering all main speakers' content and perspectives. Please:

1. Identify and summarize each main speaker's key points and contributions
2. Describe the topics, issues, or events they discussed
3. Note any important decisions, conclusions, or action items
4. Include person names mentioned
5. Use natural language, avoid too formal meeting tone
6. Fix common typos and similar-sounding words

---
{good_transcript}
"""
    response_long_summary = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a meeting transcript summarization assistant."},
            {"role": "user", "content": long_summary_prompt},
        ],
        temperature=0.2,
        timeout=120  # 2 minute timeout
    )
    long_summary = response_long_summary.choices[0].message.content
    if long_summary is None:
        log("Warning: API returned None for summary, using fallback")
        long_summary = "無法生成摘要" if language.startswith('zh') else "Unable to generate summary"
    else:
        long_summary = long_summary.strip()
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(long_summary)
    log(f"重點摘要: {summary_path}" if language.startswith('zh') else f"Summary: {summary_path}")
    return long_summary

def generate_filename_summary(long_summary):
    # Detect language and create appropriate prompt
    language = detect_language(long_summary)
    
    if language.startswith('zh'):
        summary_prompt = f"""根據下面的摘要，請給我一句話摘要，適合作為檔案名稱（盡量包含主題、重要事件或主要speaker的名字），請保持在10個字以內，不要包含任何前綴，只需主題內容：
---
{long_summary}
"""
    else:
        # English and other languages
        summary_prompt = f"""Based on the following summary, give me a one-sentence summary suitable as a filename (include the topic, important events, or names of main speakers). Keep it within 10 words, no prefixes, just the topic content:
---
{long_summary}
"""
    response_summary = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that generates concise file names from transcripts."},
            {"role": "user", "content": summary_prompt},
        ],
        temperature=0.2,
        timeout=60  # 1 minute timeout
    )
    summary = response_summary.choices[0].message.content
    if summary is None:
        log("Warning: API returned None for filename summary, using fallback")
        return "談話記錄" if language.startswith('zh') else "conversation"
    summary = summary.strip()
    summary = re.sub(r'[\\/*?:"<>|\n\r,]', '', summary)
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
    valid_options = {
        '--rename', '--force', '--verbose', '--no-refine', '--summary', '--lang', '--help', '-h', '--long_summary_prompt'
    }
    # Check for unknown options
    unknown_options = [arg for arg in sys.argv[1:] if arg.startswith('--') and not any(arg.startswith(opt) for opt in valid_options)]
    if unknown_options:
        print(f"Error: Unknown option(s): {' '.join(unknown_options)}\n")
        print("Valid options are:")
        print("  --rename [PREFIX]: Auto-rename files and generate summary for filename")
        print("  --force: Overwrite existing output files")
        print("  --verbose: Show detailed progress and real-time output")
        print("  --no-refine: Skip transcript refinement (much faster, avoids timeout issues)")
        print("  --summary: Generate conversation summary (slower but more complete)")
        print("  --lang LANGUAGE: Specify language (default: en, options: zh, ja, ko, fr, de, es, it, pt, ru)")
        print("  --long_summary_prompt FILE: Use a custom prompt file for long summary generation (transcript will be included automatically)")
        print("  --help, -h: Show this help message")
        sys.exit(1)

    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python this_script.py input_file.mov|mp4|mp3|wav [--rename [PREFIX] --force --verbose --no-refine --summary --lang LANGUAGE --long_summary_prompt FILE]")
        print("  --no-refine: Skip transcript refinement (much faster, avoids timeout issues)")
        print("  --summary: Generate conversation summary (slower but more complete)")
        print("  --rename [PREFIX]: Auto-rename files and generate summary for filename")
        print("                    PREFIX is optional (e.g., --rename AI_Panel_Discussion)")
        print("  --lang LANGUAGE: Specify language (default: en, options: zh, ja, ko, fr, de, es, it, pt, ru)")
        print("  --long_summary_prompt FILE: Use a custom prompt file for long summary generation (transcript will be included automatically)")
        print("  Note: English is used by default. Use --lang to specify other languages.")
        print("  Examples:")
        print("    python script.py video.mp4  # Uses English (default)")
        print("    python script.py video.mp4 --lang zh")
        print("    python script.py video.mp4 --lang ja --summary")
        print("    python script.py video.mp4 --lang en --summary --verbose")
        print("    python script.py video.mp4 --rename  # Auto-rename with summary")
        print("    python script.py video.mp4 --rename Interview_Vertex  # With custom prefix")
        print("    python script.py video.mp4 --long_summary_prompt custom_prompt.txt")
        sys.exit(0)

    input_file = sys.argv[1]
    rename_file = '--rename' in sys.argv
    rename_prefix = None
    
    # Parse rename prefix if provided
    if '--rename' in sys.argv:
        rename_index = sys.argv.index('--rename')
        if rename_index + 1 < len(sys.argv) and not sys.argv[rename_index + 1].startswith('--'):
            rename_prefix = sys.argv[rename_index + 1]
            log(f"Using rename prefix: {rename_prefix}")
    
    verbose = '--verbose' in sys.argv
    force = '--force' in sys.argv
    skip_refinement = '--no-refine' in sys.argv
    should_generate_summary = '--summary' in sys.argv or rename_file  # Generate summary if --summary or --rename is used
    
    # Parse language argument
    language = 'en'  # Default to English
    if '--lang' in sys.argv:
        lang_index = sys.argv.index('--lang')
        if lang_index + 1 < len(sys.argv):
            language = sys.argv[lang_index + 1]
            log(f"Using specified language: {language}")
        else:
            log("Warning: --lang specified but no language given, using English (default)")
    else:
        print("\n" + "="*60)
        print("LANGUAGE SETTING")
        print("="*60)
        print("Using English as default language.")
        print("To use other languages, specify with --lang option:")
        print("  --lang zh (Chinese)  --lang ja (Japanese)  --lang ko (Korean)")
        print("  --lang fr (French)   --lang de (German)    --lang es (Spanish)")
        print("  --lang it (Italian)  --lang pt (Portuguese) --lang ru (Russian)")
        print("="*60)
        log("Using English as default language. Use --lang to specify other languages (zh, ja, ko, fr, de, es, it, pt, ru)")

    # Parse long_summary_prompt file if provided
    long_summary_prompt_file = None
    custom_prompt_preview = None
    if '--long_summary_prompt' in sys.argv:
        lsp_index = sys.argv.index('--long_summary_prompt')
        if lsp_index + 1 < len(sys.argv) and not sys.argv[lsp_index + 1].startswith('--'):
            long_summary_prompt_file = sys.argv[lsp_index + 1]
            if not os.path.exists(long_summary_prompt_file):
                print(f"Error: Custom long summary prompt file '{long_summary_prompt_file}' not found.")
                sys.exit(1)
            # Print the file name and prompt preview at the beginning
            with open(long_summary_prompt_file, 'r', encoding='utf-8') as f:
                custom_prompt = f.read()
            print("\n==============================")
            print(f"Using custom long summary prompt file: {long_summary_prompt_file}")
            print("------------------------------")
            if '{good_transcript}' in custom_prompt:
                preview = custom_prompt.replace('{good_transcript}', '[TRANSCRIPT HERE]')
            else:
                preview = custom_prompt.rstrip() + '\n\nTranscript:\n[TRANSCRIPT HERE]'
            print(preview)
            print("==============================\n")
        else:
            print("Error: --long_summary_prompt specified but no file given.")
            sys.exit(1)

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)

    base = os.path.splitext(os.path.basename(input_file))[0]
    if base.endswith('_480p'):
        base = base[:-6]
    basepath = os.path.dirname(input_file)
    output_prefix = base

    # Add language code to output file names, but avoid duplicate lang codes
    # Use 'language' directly for file naming
    def add_land_index(base, language, ext):
        if base.endswith(f"_{language}"):
            return f"{base}{ext}"
        else:
            return f"{base}_{language}{ext}"

    raw_transcript_path = os.path.join(basepath, add_land_index(output_prefix + ".raw_transcript", language, ".txt"))
    good_transcript_path = os.path.join(basepath, add_land_index(output_prefix + ".refined_transcript", language, ".txt"))
    summary_path = os.path.join(basepath, add_land_index(output_prefix + ".summary", language, ".txt"))
    srt_path = os.path.join(basepath, add_land_index(output_prefix, language, ".srt"))

    # Only exit early if we're not doing rename and files exist
    if os.path.exists(good_transcript_path) and not force and not rename_file:
        if should_generate_summary and os.path.exists(summary_path):
            log("Both refined transcript and summary already exist. Use --force to overwrite or --rename to rename existing files.")
            sys.exit(0)
        elif not should_generate_summary:
            log("Refined transcript already exists. Use --force to overwrite or --rename to rename existing files.")
            sys.exit(0)

    # ---- STEP 1: get or generate good_transcript ----
    if os.path.exists(good_transcript_path) and not force:
        with open(good_transcript_path, "r", encoding="utf-8") as f:
            good_transcript = f.read()
        log(f"Found existing refined transcript at {good_transcript_path}")
    else:
        pipeline = load_diarization_pipeline(HF_TOKEN)
        transcript = load_or_generate_transcript(input_file, raw_transcript_path, pipeline, verbose, language, force=force)
        
        if skip_refinement:
            log("Skipping transcript refinement (--no-refine flag used)")
            good_transcript = transcript
            with open(good_transcript_path, 'w', encoding='utf-8') as f:
                f.write(good_transcript)
        else:
            good_transcript = clean_transcript(transcript, good_transcript_path)

    # ---- STEP 2: get or generate summary ----
    if should_generate_summary:
        if os.path.exists(summary_path) and not force:
            log(f"Found existing summary at {summary_path}")
            with open(summary_path, "r", encoding="utf-8") as f:
                long_summary = f.read()
        else:
            long_summary = generate_summary(good_transcript, summary_path, long_summary_prompt_file)
    else:
        log("Skipping summary generation (use --summary or --rename flag to enable)")
        long_summary = "No summary generated. Use --summary or --rename flag to generate conversation summary."

    # Save .srt subtitles
    if skip_refinement:
        if os.path.exists(raw_transcript_path):
            with open(raw_transcript_path, "r", encoding="utf-8") as f:
                transcript_lines = f.read().strip().splitlines()
            write_srt(transcript_lines, srt_path)
    else:
        if os.path.exists(good_transcript_path):
            with open(good_transcript_path, "r", encoding="utf-8") as f:
                transcript_lines = f.read().strip().splitlines()
            write_srt(transcript_lines, srt_path)

    # --- Summary-based renaming ---
    if rename_file:
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', base)
        if date_match:
            date_str = date_match.group(1)
        else:
            date_match = re.search(r'(\d{8})', base)
            if date_match:
                d = date_match.group(1)
                date_str = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            else:
                date_str = datetime.now().strftime('%Y-%m-%d')
        if should_generate_summary:
            summary_for_name = generate_filename_summary(long_summary)
        else:
            summary_for_name = "conversation"
        ext = os.path.splitext(input_file)[1]
        
        # Build the new filename with optional prefix
        if rename_prefix:
            new_base = f"{date_str}_{rename_prefix}_{summary_for_name}"
        else:
            new_base = f"{date_str}_{summary_for_name}"

        # Replace spaces with underscores in the new base name
        new_base = new_base.replace(' ', '_')

        # --- Rename main media file ---
        # Insert lang_code before the extension
        new_file_path = os.path.join(basepath, f"{new_base}_{language}{ext}")
        if os.path.abspath(new_file_path) != os.path.abspath(input_file):
            if os.path.exists(new_file_path):
                log(f"Target file {new_file_path} already exists. Skipping rename.")
            else:
                os.rename(input_file, new_file_path)
                log(f"Original file: {input_file}")
                log(f"Renamed original file to: {new_file_path}")

        # --- Rename transcript and summary files ---
        new_raw_transcript_path = os.path.join(basepath, add_land_index(new_base + ".raw_transcript", language, ".txt"))
        new_good_transcript_path = os.path.join(basepath, add_land_index(new_base + ".refined_transcript", language, ".txt"))
        new_summary_path = os.path.join(basepath, add_land_index(new_base + ".summary", language, ".txt"))
        new_srt_path = os.path.join(basepath, add_land_index(new_base, language, ".srt"))
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
        if should_generate_summary:
            safe_rename(summary_path, new_summary_path)
        safe_rename(srt_path, new_srt_path)

if __name__ == "__main__":
    main()
