# Transcription and Summarization Project

A comprehensive Python toolkit for transcribing audio/video files, generating summaries, and managing speaker diarization with multilingual support.

## Features

- **Speaker Diarization**: Automatically identifies and separates different speakers
- **Multilingual Support**: Chinese, English, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian
- **Language Support**: English by default, other languages via --lang option
- **Batch Processing**: Process multiple files efficiently
- **Smart File Renaming**: Auto-rename files based on content summary
- **SRT Subtitle Generation**: Create subtitle files for videos
- **Transcript Cleaning**: AI-powered transcript refinement
- **Detailed Logging**: Verbose mode for debugging and monitoring

## Prerequisites

### Required Software
- **FFmpeg**: For audio/video conversion
- **Python 3.8+**: For running the scripts

### Python Dependencies
```bash
pip install openai pyannote.audio pydub torch pywhispercpp python-dotenv
```

### Environment Setup
1. Create a `.env` file in the project directory:
```bash
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

2. Get your HuggingFace token from [HuggingFace](https://huggingface.co/settings/tokens)
3. Get your OpenAI API key from [OpenAI](https://platform.openai.com/api-keys)

## Scripts Overview

### Main Script: `transcribe_and_summarize_recording_zhOnly_assign.speaker_optionaly.rename.py`

The primary script for processing individual audio/video files.

**Usage:**
```bash
python transcribe_and_summarize_recording_zhOnly_assign.speaker_optionaly.rename.py input_file [OPTIONS]
```

**Options:**
- `--rename`: Auto-rename files based on content summary
- `--force`: Overwrite existing output files
- `--verbose`: Show detailed progress and real-time output
- `--no-clean`: Skip transcript cleaning (faster processing)
- `--lang LANGUAGE`: Specify language (zh, en, ja, ko, fr, de, es, it, pt, ru)

**Examples:**
```bash
# Basic usage (English default)
python transcribe_and_summarize_recording_zhOnly_assign.speaker_optionaly.rename.py video.mp4

# Chinese transcription
python transcribe_and_summarize_recording_zhOnly_assign.speaker_optionaly.rename.py video.mp4 --lang zh

# With verbose output and file renaming
python transcribe_and_summarize_recording_zhOnly_assign.speaker_optionaly.rename.py video.mp4 --lang zh --verbose --rename

# Fast mode (skip cleaning)
python transcribe_and_summarize_recording_zhOnly_assign.speaker_optionaly.rename.py video.mp4 --no-clean
```

### Batch Processing: `batch_transcribe.py`

Process multiple files in a directory efficiently.

**Usage:**
```bash
python batch_transcribe.py [TARGET_DIRECTORY] [OPTIONS]
```

**Options:**
- `--no-clean`: Skip transcript cleaning (faster)
- `--verbose`: Show detailed progress from main script
- `--force`: Overwrite existing output files
- `--lang LANG`: Specify language for all files
- `--help, -h`: Show help message

**Examples:**
```bash
# Process current directory (English default)
python batch_transcribe.py --verbose

# Process specific directory (Chinese)
python batch_transcribe.py ~/Videos --lang zh --no-clean

# Process with verbose output (Japanese)
python batch_transcribe.py . --lang ja --verbose --force
```


## Output Files

For each input file, the script generates:

1. **Raw Transcript** (`*.gpu.speakers.raw_transcript.txt`): Original transcription with speaker labels
2. **Clean Transcript** (`*.gpu.speakers.clean_transcript.txt`): AI-refined transcript
3. **Summary** (`*.gpu.speakers.summary.txt`): Content summary (up to 1000 words)
4. **SRT Subtitles** (`*.srt`): Subtitle file for video players
5. **Renamed Files** (if `--rename` used): Original file renamed with date and summary

## Language Support

### Supported Languages
- **zh**: Chinese (Traditional/Simplified)
- **en**: English
- **ja**: Japanese
- **ko**: Korean
- **fr**: French
- **de**: German
- **es**: Spanish
- **it**: Italian
- **pt**: Portuguese
- **ru**: Russian
- **auto**: Auto-detection (available in language prompt, but not recommended)

### Language Selection
- **Default**: English is used if no language is specified
- **Optional**: Use `--lang` option to specify other languages
- Supported languages: zh, ja, ko, fr, de, es, it, pt, ru
- For batch processing, `--lang` is optional (defaults to English)

## Processing Pipeline

1. **Audio Conversion**: Convert to mono 16kHz WAV format
2. **Speaker Diarization**: Identify and separate speakers
3. **Transcription**: Generate raw transcript with timestamps
4. **Language Selection**: Use specified language or default to English
5. **Transcript Cleaning**: AI-powered refinement and formatting
6. **Summary Generation**: Create content summary
7. **File Renaming**: Optionally rename based on summary
8. **SRT Generation**: Create subtitle file

## Performance Tips

### Speed Optimization
- Use `--no-clean` for faster processing (skips transcript refinement)
- Use smaller Whisper models for faster transcription
- Process files in parallel using batch script

### Quality Optimization
- Use `--verbose` to monitor progress and catch issues
- Specify language with `--lang` for non-English content
- Use `--force` to regenerate existing outputs

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Install FFmpeg
   brew install ffmpeg  # macOS
   sudo apt install ffmpeg  # Ubuntu
   ```

2. **Missing API keys**
   - Ensure `.env` file exists with valid tokens
   - Check token permissions and quotas

3. **Memory issues with large files**
   - Use `--no-clean` to reduce memory usage
   - Process files individually instead of batch

4. **Timeout errors**
   - Use `--no-clean` for faster processing
   - Check internet connection for API calls

### Log Files
- Batch processing creates `batch_transcribe.log` in current directory
- Check logs for detailed error information

## File Formats

### Input Formats
- **Video**: .mov, .mp4
- **Audio**: .mp3, .wav, .m4a

### Output Formats
- **Text**: .txt (UTF-8 encoding)
- **Subtitles**: .srt
- **Logs**: .log

## Configuration

### Environment Variables
```bash
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

### Model Settings
- **Whisper Model**: `large-v3` (default)
- **Speaker Diarization**: `pyannote/speaker-diarization-3.1`
- **Summary Model**: `gpt-4o`
- **Cleaning Model**: `gpt-4.1-mini`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for personal use. Please respect the terms of service for OpenAI and HuggingFace APIs.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Ensure all dependencies are installed
4. Verify API keys and permissions 