# Batch Transcription Script

This directory contains a batch processing script for transcribing multiple video files.

## `batch_transcribe.py` (Updated)

A wrapper script that processes multiple files using the main transcription script. **Now works from any location!**

### Features:
- ✅ **Works from any directory** - No need to be in the same folder as video files
- ✅ Processes both `.mov` and `.mp4` files
- ✅ Supports all command line options from the main script
- ✅ Detailed logging to `batch_transcribe.log` in the target directory
- ✅ Progress tracking and timing
- ✅ Error handling and recovery
- ✅ Speaker diarization support
- ✅ Timeout handling (12 hours per file)
- ✅ Success/failure counting

### Usage:

#### **From Any Location:**
```bash
# Process files in current directory
python3 ~/projects/transcrib_and_summary/batch_transcribe.py

# Process files in a specific directory
python3 ~/projects/transcrib_and_summary/batch_transcribe.py /path/to/videos

# Process files in current directory with options
python3 ~/projects/transcrib_and_summary/batch_transcribe.py . --no-clean --verbose

# Process files in specific directory with language specification
python3 ~/projects/transcrib_and_summary/batch_transcribe.py /path/to/videos --lang zh --verbose
```

#### **Command Line Options:**
- `--no-clean`: Skip transcript cleaning (much faster)
- `--verbose`: Show detailed progress
- `--force`: Overwrite existing output files
- `--lang LANGUAGE`: Specify language (e.g., zh, en, ja, ko, fr, de, etc.)

### Examples:

```bash
# Basic usage - process current directory
python3 ~/projects/transcrib_and_summary/batch_transcribe.py

# Process specific folder
python3 ~/projects/transcrib_and_summary/batch_transcribe.py ~/Videos/recordings

# Fast processing with Chinese language
python3 ~/projects/transcrib_and_summary/batch_transcribe.py . --no-clean --lang zh --verbose

# High-quality processing with English language
python3 ~/projects/transcrib_and_summary/batch_transcribe.py /path/to/videos --lang en --verbose

# Force overwrite existing files
python3 ~/projects/transcrib_and_summary/batch_transcribe.py . --force --verbose
```

### Output:
- Logs all operations to `batch_transcribe.log` in the target directory
- Shows progress: `(1/5) Processing: video1.mp4`
- Reports success/failure counts
- Shows processing time for each file

### File Structure:
```
~/projects/transcrib_and_summary/
├── batch_transcribe.py
├── transcribe_and_summarize_recording_zhOnly_assign.speaker_optionaly.rename.py
└── README_batch.md

~/Videos/recordings/  (or any other directory)
├── video1.mp4
├── video2.mov
├── batch_transcribe.log  (created here)
├── video1.gpu.speakers.raw_transcript.txt
├── video1.gpu.speakers.clean_transcript.txt
├── video1.gpu.speakers.summary.txt
└── video1.srt
```

## Troubleshooting

### Common Issues:

1. **Script not found:**
   - Ensure both scripts are in the same directory (`~/projects/transcrib_and_summary/`)
   - Check the script path in the error message

2. **Target directory not found:**
   - Verify the directory path exists
   - Use absolute paths if needed

3. **Timeout errors:**
   - Use `--no-clean` for faster processing
   - Check internet connection
   - Monitor `batch_transcribe.log` for details

4. **Permission errors:**
   - Ensure you have write permissions in the target directory
   - Check disk space

5. **FFmpeg not found:**
   - Install FFmpeg: `brew install ffmpeg`

### Log File:
Check `batch_transcribe.log` in the target directory for detailed error information and processing status. 