import os
import re
import subprocess
import sys
from datetime import datetime

import openai
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment
from pywhispercpp.model import Model

# Load .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    print("Error: HF_TOKEN is not set. Please check your .env file.")
    sys.exit(1)

# Start timer
start_time = datetime.now()


def elapsed():
    """Return elapsed time since script start."""
    delta = datetime.now() - start_time
    return f"{delta.total_seconds():.1f}s"


def log(msg):
    """Log message with timestamp and elapsed time."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] (+{elapsed()}) {msg}")


def ensure_wav_mono_16k(input_path):
    """Convert audio file to mono 16kHz WAV format."""
    import wave

    base, ext = os.path.splitext(input_path)
    out_wav = base + "_16k_mono.wav"

    # Check if input is already the right format
    if ext.lower() == ".wav":
        try:
            with wave.open(input_path, 'rb') as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == 16000:
                    return input_path
        except Exception as e:
            log(f"Warning: Could not check WAV format: {e}")

    # Convert to required format
    log(f"Converting {input_path} to mono 16kHz WAV...")
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, '-ar', '16000', '-ac', '1', out_wav],
            check=True, capture_output=True, text=True
        )
        return out_wav
    except subprocess.CalledProcessError as e:
        log(f"FFmpeg conversion failed: {e}")
        log(f"FFmpeg stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        log("Error: FFmpeg not found. Please install FFmpeg.")
        raise


def load_diarization_pipeline(token):
    """Load and configure speaker diarization pipeline."""
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    )
    pipeline.to(device)
    return pipeline


def run_diarization_and_transcription(
    audio_file, pipeline, whisper_model_path="large-v3", verbose=False, language=None
):
    """Run speaker diarization and transcription on audio file."""
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


def load_or_generate_transcript(input_file, raw_transcript_path, pipeline, verbose=False, language=None):
    """Load existing transcript or generate new one."""
    if os.path.exists(raw_transcript_path):
        log(f"Found existing raw transcript at {raw_transcript_path}, skipping audio conversion, diarization, and transcription.")
        with open(raw_transcript_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        audio_file = ensure_wav_mono_16k(input_file)
        transcript = run_diarization_and_transcription(
            audio_file, pipeline, verbose=verbose, language=language
        )
        with open(raw_transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        if audio_file.endswith("_16k_mono.wav") and os.path.exists(audio_file):
            os.remove(audio_file)
            log(f"Deleted temporary file: {audio_file}")
        return transcript


def clean_transcript(transcript, good_transcript_path):
    """Clean and refine transcript using AI."""
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
    log(f"修飾逐字稿: {good_transcript_path}")
    return good_transcript


def detect_language(text):
    """Simple language detection based on character sets."""
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
    """Clean a single chunk of transcript using AI."""
    # Detect language and create appropriate prompt
    language = detect_language(transcript_chunk)

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
                    {"role": "system", "content": "你是一個優秀的中文逐字稿修飾助手。"},
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


def generate_summary(good_transcript, summary_path):
    """Generate summary from cleaned transcript."""
    # Detect language and create appropriate prompt
    language = detect_language(good_transcript)

    if language.startswith('zh'):
        long_summary_prompt = f"""請根據下面的的逐字稿(可能是與心理師談話)，以主要speaker的內容寫一段1000字以內的摘要，講述她最近的生活狀態，請把人物名稱標注在內，用字自然，不要有開會的感覺，修正常見錯別字、類似音的字 例如：產修、殘修 其實都是禪修），繁體中文：

---
{good_transcript}
"""
    else:
        # English and other languages
        long_summary_prompt = f"""Based on the following transcript (possibly a therapy session), write a summary of up to 1000 words about the main speaker's recent life situation. Include character names, use natural language, avoid meeting-like tone, fix common typos and similar-sounding words:

---
{good_transcript}
"""

    # Try gpt-4.1-mini with increasing timeouts and retries
    timeouts_to_try = [180, 240, 300]  # 3min, 4min, 5min
    max_retries = 3

    for attempt in range(max_retries):
        timeout_seconds = timeouts_to_try[attempt] if attempt < len(timeouts_to_try) else timeouts_to_try[-1]

        try:
            log(f"Attempt {attempt + 1}/{max_retries}: Trying gpt-4.1-mini with {timeout_seconds}s timeout...")
            response_summary = openai.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "你是一個優秀的中文摘要助手。"},
                    {"role": "user", "content": long_summary_prompt},
                ],
                temperature=0.3,
                timeout=timeout_seconds
            )
            long_summary = response_summary.choices[0].message.content
            if long_summary is None:
                log(f"Warning: gpt-4.1-mini returned None on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    continue
                else:
                    break
            log(f"Successfully generated summary using gpt-4.1-mini (attempt {attempt + 1})")
            break

        except Exception as e:
            log(f"Error with gpt-4.1-mini (attempt {attempt + 1}): {e}")
            if "timeout" in str(e).lower():
                log(f"Timeout with gpt-4.1-mini on attempt {attempt + 1}, trying again with longer timeout...")
            if attempt < max_retries - 1:
                continue
            else:
                break
    else:
        # If all attempts fail, create a simple summary
        log("All attempts with gpt-4.1-mini failed for summary generation, creating simple summary")
        long_summary = f"Summary of transcript with {len(good_transcript)} characters."

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(long_summary or "")
    log(f"摘要: {summary_path}")
    return long_summary or ""


def generate_filename_summary(long_summary):
    """Generate short filename from long summary."""
    # Detect language and create appropriate prompt
    language = detect_language(long_summary)

    if language.startswith('zh'):
        filename_prompt = f"""請根據下面的摘要，生成一個簡短的中文檔名（不超過20個字），用於重命名錄音檔案：

---
{long_summary}
"""
    else:
        # English and other languages
        filename_prompt = f"""Based on the following summary, generate a short English filename (no more than 20 words) for renaming the recording file:

---
{long_summary}
"""

    try:
        response_filename = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "你是一個優秀的檔名生成助手。"},
                {"role": "user", "content": filename_prompt},
            ],
            temperature=0.3,
            timeout=60
        )
        filename_summary = response_filename.choices[0].message.content
        if filename_summary is None:
            log("Warning: gpt-4.1-mini returned None for filename generation")
            return "conversation"
        return filename_summary.strip() or "conversation"
    except Exception as e:
        log(f"Error generating filename: {e}")
        return "conversation"


def write_srt(transcript_lines, srt_path):
    """Write transcript to SRT format."""
    def format_timestamp(seconds):
        """Format seconds to SRT timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(transcript_lines, 1):
            # Extract timestamp and text from line
            match = re.search(r'\[(\d+\.\d+)-(\d+\.\d+)\] (.+)', line)
            if match:
                start_time, end_time, text = match.groups()
                f.write(f"{i}\n")
                f.write(f"{format_timestamp(float(start_time))} --> {format_timestamp(float(end_time))}\n")
                f.write(f"{text}\n\n")


def main():
    """Main function to process audio files."""
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe and summarize audio recordings")
    parser.add_argument("input_file", help="Input audio file path")
    parser.add_argument("--lang", choices=["en", "zh"], default="en", help="Language for transcription (default: en)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-refine", action="store_true", help="Skip transcript refinement")
    parser.add_argument("--summary", action="store_true", help="Generate summary")
    parser.add_argument("--rename", help="Custom prefix for file renaming")

    args = parser.parse_args()

    # Set OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Error: OPENAI_API_KEY is not set. Please check your .env file.")
        sys.exit(1)

    # Auto-enable summary if rename is specified
    if args.rename:
        args.summary = True

    # Determine language
    language = args.lang
    if language == "en":
        language = "en"
    elif language == "zh":
        language = "zh"
    else:
        language = "en"  # Default to English

    # Create output directory
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = f"output_{base_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    raw_transcript_path = os.path.join(output_dir, "raw_transcript.txt")
    good_transcript_path = os.path.join(output_dir, "refined_transcript.txt")
    summary_path = os.path.join(output_dir, "summary.txt")
    srt_path = os.path.join(output_dir, "transcript.srt")

    log(f"Processing: {args.input_file}")
    log(f"Language: {language}")
    log(f"Output directory: {output_dir}")

    try:
        # Load diarization pipeline
        pipeline = load_diarization_pipeline(HF_TOKEN)

        # Load or generate transcript
        transcript = load_or_generate_transcript(
            args.input_file, raw_transcript_path, pipeline, verbose=args.verbose, language=language
        )

        # Parse transcript lines for SRT generation
        transcript_lines = [line.strip() for line in transcript.split('\n') if line.strip()]

        # Write SRT file
        write_srt(transcript_lines, srt_path)
        log(f"SRT file: {srt_path}")

        # Clean transcript if not skipped
        if not args.no_refine:
            good_transcript = clean_transcript(transcript, good_transcript_path)
        else:
            log("Skipping transcript refinement")
            good_transcript = transcript
            with open(good_transcript_path, 'w', encoding='utf-8') as f:
                f.write(good_transcript)

        # Generate summary if requested
        if args.summary:
            long_summary = generate_summary(good_transcript, summary_path)
            
            # Generate filename if rename is requested
            if args.rename:
                filename_summary = generate_filename_summary(long_summary)
                # Clean filename for filesystem
                safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename_summary)
                safe_filename = safe_filename.replace(' ', '_')
                
                # Create new filename with custom prefix
                new_filename = f"{args.rename}_{safe_filename}"
                
                # Rename output directory
                new_output_dir = f"output_{new_filename}"
                if os.path.exists(new_output_dir):
                    log(f"Warning: Output directory {new_output_dir} already exists")
                else:
                    try:
                        os.rename(output_dir, new_output_dir)
                        log(f"Renamed output directory to: {new_output_dir}")
                        output_dir = new_output_dir
                    except OSError as e:
                        log(f"Warning: Could not rename directory: {e}")

        log("Processing completed successfully!")

    except Exception as e:
        log(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
