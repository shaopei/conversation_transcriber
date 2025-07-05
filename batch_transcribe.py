import os
import subprocess
import glob
from datetime import datetime
import time

SCRIPT = "transcribe_and_summarize_recording_zhOnly_assign.speaker_optionaly.rename.py"
LOGFILE = "batch_transcribe.log"

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {msg}\n")
    print(f"[{now}] {msg}")

def check_script_exists():
    if not os.path.exists(SCRIPT):
        log(f"ERROR: Script {SCRIPT} not found!")
        return False
    return True

def main():
    if not check_script_exists():
        return
    
    folder = "."
    files = sorted(glob.glob(os.path.join(folder, "*.mov")) + glob.glob(os.path.join(folder, "*.mp4")))
    
    if not files:
        log("No .mov or .mp4 files found in current directory.")
        return
    
    log(f"Batch processing started. Found {len(files)} files.")
    
    # Check for command line options
    import sys
    args = []
    if "--no-clean" in sys.argv:
        args.append("--no-clean")
        log("Using --no-clean mode (faster processing)")
    if "--verbose" in sys.argv:
        args.append("--verbose")
        log("Using --verbose mode (detailed output)")
    if "--force" in sys.argv:
        args.append("--force")
        log("Using --force mode (overwrite existing files)")

    successful = 0
    failed = 0
    
    for i, f in enumerate(files, 1):
        log(f"({i}/{len(files)}) Processing: {f}")
        start_time = time.time()
        
        try:
            # Build command with arguments
            cmd = ["python3", SCRIPT, f] + args
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=12*3600  # 12h timeout in case of very large file
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                log(f"SUCCESS: {f} (took {elapsed:.1f}s)")
                successful += 1
            else:
                log(f"FAIL: {f} (took {elapsed:.1f}s)")
                log(f"Return code: {result.returncode}")
                if result.stdout:
                    log(f"STDOUT: {result.stdout}")
                if result.stderr:
                    log(f"STDERR: {result.stderr}")
                failed += 1
                
        except subprocess.TimeoutExpired:
            log(f"TIMEOUT: {f} (exceeded 12 hours)")
            failed += 1
        except Exception as e:
            log(f"EXCEPTION: {f}: {e}")
            failed += 1
    
    log(f"Batch processing complete. Success: {successful}, Failed: {failed}")
    
    if failed > 0:
        log("Some files failed. Check the log above for details.")
        log("You can retry failed files individually or run with --force to overwrite existing outputs.")

if __name__ == "__main__":
    main()
