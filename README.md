# Batch Transcription Script

This directory contains a batch processing script for transcribing multiple video files.

## `batch_transcribe.py` (Recommended)

A wrapper script that processes multiple files using the main transcription script.

### Features:
- ✅ Processes both `.mov` and `.mp4` files
- ✅ Supports all command line options from the main script
- ✅ Detailed logging to `batch_transcribe.log`
- ✅ Progress tracking and timing
- ✅ Error handling and recovery
- ✅ Speaker diarization support
- ✅ Timeout handling (12 hours per file)
- ✅ Success/failure counting

### Usage:
```bash
# Basic batch processing
python3 batch_transcribe.py

# Fast processing (skip transcript cleaning)
python3 batch_transcribe.py --no-clean

# Verbose output
python3 batch_transcribe.py --verbose

# Force overwrite existing files
python3 batch_transcribe.py --force

# Combine options
python3 batch_transcribe.py --no-clean --verbose
```

### Command Line Options:
- `--no-clean`: Skip transcript cleaning (much faster)
- `--verbose`: Show detailed progress
- `--force`: Overwrite existing output files

### Output:
- Logs all operations to `batch_transcribe.log`
- Shows progress: `(1/5) Processing: video1.mp4`
- Reports success/failure counts
- Shows processing time for each file

## Recommended Workflow

1. **For quick processing:**
   ```bash
   python3 batch_transcribe.py --no-clean --verbose
   ```

2. **For high-quality results:**
   ```bash
   python3 batch_transcribe.py --verbose
   ```

3. **For debugging issues:**
   ```bash
   python3 batch_transcribe.py --verbose --force
   ```

## Troubleshooting

### Common Issues:

1. **Script not found:**
   - Ensure `transcribe_and_summarize_recording_zhOnly_assign.speaker_optionaly.rename.py` is in the same directory

2. **Timeout errors:**
   - Use `--no-clean` for faster processing
   - Check internet connection
   - Monitor `batch_transcribe.log` for details

3. **Permission errors:**
   - Ensure you have write permissions in the directory
   - Check disk space

4. **FFmpeg not found:**
   - Install FFmpeg: `brew install ffmpeg`

### Log File:
Check `batch_transcribe.log` for detailed error information and processing status. 