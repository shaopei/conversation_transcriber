# Conversation Transcriber

Transform your conversations into organized, searchable transcripts with AI-powered speaker separation and intelligent summarization. Perfect for meetings, interviews, therapy sessions, podcasts, and any multi-speaker recordings.

## ✨ Key Features

- **🎤 Smart Speaker Separation**: Automatically identifies and labels different speakers in conversations
- **🌍 Multilingual Conversations**: Support for 10+ languages including Chinese, English, Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian
- **🤖 AI-Powered Cleaning**: Intelligent transcript refinement that removes filler words and fixes common errors
- **📝 Intelligent Summaries**: Generate comprehensive summaries of conversations (up to 1000 words)
- **🎬 Subtitle Generation**: Create SRT subtitle files for video players
- **📁 Smart File Organization**: Auto-rename files based on conversation content
- **⚡ Batch Processing**: Process multiple conversation files efficiently
- **🔍 Detailed Progress Tracking**: Real-time monitoring with verbose mode

## 🚀 Quick Start

### What You'll Get
- **Speaker-labeled transcripts** with timestamps
- **Clean, readable text** with proper punctuation
- **Intelligent summaries** capturing key points
- **Searchable conversation records** for easy reference

### Perfect For
- **Business Meetings**: Capture action items and decisions
- **Interviews**: Document Q&A sessions with speaker clarity
- **Therapy Sessions**: Maintain detailed session records
- **Podcasts**: Create show notes and transcripts
- **Academic Discussions**: Document research conversations
- **Legal Proceedings**: Maintain accurate conversation records

## 📋 Requirements

### Software Requirements
- **FFmpeg**: For audio/video conversion
- **Python 3.8+**: For running the scripts

### Python Dependencies
```bash
pip install openai pyannote.audio pydub torch pywhispercpp python-dotenv
```

### API Keys Setup
1. Create a `.env` file in the project directory:
```bash
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

2. Get your HuggingFace token from [HuggingFace](https://huggingface.co/settings/tokens)
3. Get your OpenAI API key from [OpenAI](https://platform.openai.com/api-keys)

## 🛠️ Usage Guide

### Single Conversation Processing

Process one conversation file at a time with full control over options.

**Usage:**
```bash
python conversation_transcriber.py input_file [OPTIONS]
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
python conversation_transcriber.py video.mp4

# Chinese transcription
python conversation_transcriber.py video.mp4 --lang zh

# With verbose output and file renaming
python conversation_transcriber.py video.mp4 --lang zh --verbose --rename

# Fast mode (skip cleaning)
python conversation_transcriber.py video.mp4 --no-clean
```

### Batch Conversation Processing

Process multiple conversation files efficiently - perfect for processing entire folders of recordings.

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


## 📄 Output Files

For each conversation, you'll get organized, searchable files:

1. **🎤 Raw Transcript** (`*.gpu.speakers.raw_transcript.txt`): Original transcription with speaker labels and timestamps
2. **✨ Clean Transcript** (`*.gpu.speakers.clean_transcript.txt`): AI-refined, readable version with proper punctuation
3. **📝 Summary** (`*.gpu.speakers.summary.txt`): Intelligent summary capturing key points and decisions
4. **🎬 Subtitles** (`*.srt`): Ready-to-use subtitle files for video players
5. **📁 Renamed Files** (if `--rename` used): Original files renamed with date and conversation topic

## 🌍 Language Support

### Supported Languages
- **🇨🇳 zh**: Chinese (Traditional/Simplified)
- **🇺🇸 en**: English (default)
- **🇯🇵 ja**: Japanese
- **🇰🇷 ko**: Korean
- **🇫🇷 fr**: French
- **🇩🇪 de**: German
- **🇪🇸 es**: Spanish
- **🇮🇹 it**: Italian
- **🇵🇹 pt**: Portuguese
- **🇷🇺 ru**: Russian

### Language Selection
- **Default**: English is used automatically
- **Other Languages**: Use `--lang` option to specify
- **Batch Processing**: Works with any supported language

### Testing Status
- **✅ Fully Tested**: English, Chinese
- **⚠️ Limited Testing**: Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian
- **💡 Recommendation**: For best results, use English or Chinese. Other languages may have varying accuracy.

### Hardware Testing
- **✅ Tested**: MacBook Pro M4
- **⚠️ Untested**: Other macOS versions, Windows, Linux
- **💡 Note**: Performance may vary on different hardware configurations

## 🔄 How It Works

1. **🎵 Audio Processing**: Convert your recording to optimal format
2. **👥 Speaker Detection**: AI identifies and separates different speakers
3. **📝 Transcription**: Generate accurate text with timestamps
4. **🌍 Language Processing**: Apply language-specific optimizations
5. **✨ AI Cleaning**: Remove filler words and fix errors
6. **📋 Summarization**: Create intelligent summaries of key points
7. **📁 Organization**: Optionally rename files based on content
8. **🎬 Subtitle Creation**: Generate ready-to-use subtitle files

## ⚡ Performance Tips

### 🚀 Speed Optimization
- Use `--no-clean` for faster processing (skips AI refinement)
- Process multiple files with batch script for efficiency
- Use `--verbose` to monitor progress in real-time

### 🎯 Quality Optimization
- Specify language with `--lang` for better accuracy
- Use `--force` to regenerate existing outputs
- Monitor logs for any processing issues

## 🔧 Troubleshooting

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

### 📋 Log Files
- Batch processing creates `batch_transcribe.log` in current directory
- Check logs for detailed error information

## 📁 File Formats

### Input Formats
- **🎬 Video**: .mov, .mp4
- **🎵 Audio**: .mp3, .wav, .m4a

### Output Formats
- **📄 Text**: .txt (UTF-8 encoding)
- **🎬 Subtitles**: .srt
- **📋 Logs**: .log

## ⚙️ Configuration

### Environment Variables
```bash
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_api_key
```

### AI Models Used
- **🎤 Whisper Model**: `large-v3` (high-accuracy transcription)
- **👥 Speaker Diarization**: `pyannote/speaker-diarization-3.1` (speaker separation)
- **📝 Summary Model**: `gpt-4o` (intelligent summarization)
- **✨ Cleaning Model**: `gpt-4.1-mini` (transcript refinement)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for personal use. Please respect the terms of service for OpenAI and HuggingFace APIs.

## 💬 Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Ensure all dependencies are installed
4. Verify API keys and permissions 