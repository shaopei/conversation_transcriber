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

## 📦 Installation

### Option 1: Install from GitHub (Recommended)
```bash
pip install git+https://github.com/shaopei/conversation_transcriber.git
```

### Option 2: Install from Source
```bash
git clone https://github.com/shaopei/conversation_transcriber.git
cd conversation_transcriber
pip install -e .
```

> **Note:** This package is not published on PyPI yet. Please use the GitHub installation methods above.

## 📋 Requirements

### Software Requirements
- **FFmpeg**: For audio/video conversion
- **Python 3.8+**: For running the scripts

### API Keys Setup
1. Copy the example environment file:
```bash
cp env.example .env
```

2. Edit `.env` and add your API keys:
```bash
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

3. Get your HuggingFace token from [HuggingFace](https://huggingface.co/settings/tokens)
4. Get your OpenAI API key from [OpenAI](https://platform.openai.com/api-keys)

## 🛠️ Usage Guide

### Single Conversation Processing

Process one conversation file at a time with full control over options.

**Usage:**
```bash
# If installed as package
conversation-transcriber input_file [OPTIONS]

# Or run directly
python conversation_transcriber.py input_file [OPTIONS]
```

**Options:**
- `--rename [PREFIX]`: Auto-rename files based on content summary (includes summary generation)
- `--force`: Overwrite existing output files
- `--verbose`: Show detailed progress and real-time output
- `--no-refine`: Skip transcript refinement (faster processing)
- `--summary`: Generate conversation summary (slower but more complete)
- `--lang LANGUAGE`: Specify language (zh, en, ja, ko, fr, de, es, it, pt, ru)

**Examples:**
```bash
# Basic usage (English default, no summary)
conversation-transcriber video.mp4

# Chinese transcription with summary
conversation-transcriber video.mp4 --lang zh --summary

# Auto-rename with summary (--rename includes summary generation)
conversation-transcriber video.mp4 --rename

# Auto-rename with custom prefix
conversation-transcriber video.mp4 --rename AI_Panel_Discussion

# With verbose output and file renaming
conversation-transcriber video.mp4 --lang zh --verbose --rename

# Fast mode (skip refinement, no summary)
conversation-transcriber video.mp4 --no-refine
```

### Batch Conversation Processing

Process multiple conversation files efficiently - perfect for processing entire folders of recordings.

**Usage:**
```bash
# If installed as package
batch-transcribe [TARGET_DIRECTORY] [OPTIONS]

# Or run directly
python batch_transcribe.py [TARGET_DIRECTORY] [OPTIONS]
```

**Options:**
- `--no-refine`: Skip transcript refinement (faster)
- `--summary`: Generate conversation summaries
- `--verbose`: Show detailed progress from main script
- `--force`: Overwrite existing output files
- `--lang LANG`: Specify language for all files
- `--help, -h`: Show help message

**Examples:**
```bash
# Process current directory (English default, no summaries)
batch-transcribe --verbose

# Process specific directory (Chinese with summaries)
batch-transcribe ~/Videos --lang zh --summary

# Process with verbose output (Japanese, no refinement)
batch-transcribe . --lang ja --verbose --force --no-refine
```


## 📄 Output Files

For each conversation, you'll get organized, searchable files:

1. **🎤 Raw Transcript** (`*.gpu.speakers.raw_transcript.txt`): Original transcription with speaker labels and timestamps
2. **✨ Refined Transcript** (`*.gpu.speakers.refined_transcript.txt`): AI-refined, readable version with proper punctuation
3. **📝 Summary** (`*.gpu.speakers.summary.txt`): Intelligent summary capturing key points and decisions (only with `--summary` or `--rename`)
4. **🎬 Subtitles** (`*.srt`): Ready-to-use subtitle files for video players
5. **📁 Renamed Files** (if `--rename` used): Original files renamed with date, optional prefix, and conversation topic

## 🌍 Language Support

### Supported Languages
- **zh**: Chinese (Traditional/Simplified)
- **en**: English (default)
- **ja**: Japanese
- **ko**: Korean
- **fr**: French
- **de**: German
- **es**: Spanish
- **it**: Italian
- **pt**: Portuguese
- **ru**: Russian

### Language Selection
- **Default**: English is used automatically
- **Other Languages**: Use `--lang` option to specify
- **Batch Processing**: Works with any supported language

### Testing Status
- **✅ Fully Tested**: English, Chinese
- **⚠️ Limited Testing**: Japanese, Korean, French, German, Spanish, Italian, Portuguese, Russian
- **💡 Recommendation**: For best results, use English or Chinese. Other languages may have varying accuracy.

### Hardware Testing
- **✅ Tested**: MacBook Pro M4 (Apple Silicon with MPS GPU acceleration)
- **⚠️ Untested**: Other macOS versions, Windows, Linux
- **💡 Note**: The script supports multiple GPU types (MPS for Apple, CUDA for NVIDIA, OpenCL for AMD) but has only been tested on M4. Performance may vary on different hardware configurations.

## 🔄 How It Works

1. **🎵 Audio Processing**: Convert your recording to optimal format
2. **👥 Speaker Detection**: AI identifies and separates different speakers
3. **📝 Transcription**: Generate accurate text with timestamps
4. **🌍 Language Processing**: Apply language-specific optimizations
5. **✨ AI Refinement**: Remove filler words and fix errors
6. **📋 Summarization**: Create intelligent summaries of key points
7. **📁 Organization**: Optionally rename files based on content
8. **🎬 Subtitle Creation**: Generate ready-to-use subtitle files

## ⚡ Performance Tips

### 🚀 Speed Optimization
- Use `--no-refine` for faster processing (skips AI refinement)
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
   - Use `--no-refine` to reduce memory usage
   - Process files individually instead of batch

4. **Timeout errors**
   - Use `--no-refine` for faster processing
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
- **✨ Refinement Model**: `gpt-4.1-mini` (transcript refinement)

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