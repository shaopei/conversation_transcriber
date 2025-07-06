import os
import subprocess
import glob
from datetime import datetime
import time
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(SCRIPT_DIR, "conversation_transcriber.py")

def log(msg):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Create log file in the current working directory
    logfile = os.path.join(os.getcwd(), "batch_transcribe.log")
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"[{now}] {msg}\n")
    print(f"[{now}] {msg}")

def check_script_exists():
    if not os.path.exists(SCRIPT):
        log(f"ERROR: Script {SCRIPT} not found!")
        log(f"Expected location: {SCRIPT}")
        return False
    return True

def main():
    # Show help if requested
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python3 batch_transcribe.py [TARGET_DIRECTORY] [OPTIONS]")
        print("")
        print("TARGET_DIRECTORY:")
        print("  Directory containing video files (default: current directory)")
        print("")
        print("OPTIONS:")
        print("  --no-refine    Skip transcript refinement (faster)")
        print("  --summary      Generate conversation summaries")
        print("  --verbose      Show detailed progress")
        print("  --force        Overwrite existing output files")
        print("  --lang LANG    Specify language (default: en, options: zh, ja, ko, fr, de, es, it, pt, ru)")
        print("  --help, -h     Show this help message")
        print("")
        print("EXAMPLES:")
        print("  python3 ~/projects/transcrib_and_summary/batch_transcribe.py")
        print("  python3 ~/projects/transcrib_and_summary/batch_transcribe.py . --verbose")
        print("  python3 ~/projects/transcrib_and_summary/batch_transcribe.py ~/Videos --lang zh --no-refine")
        print("  python3 ~/projects/transcrib_and_summary/batch_transcribe.py ~/Videos --summary  # With summaries")
        print("  python3 ~/projects/transcrib_and_summary/batch_transcribe.py ~/Videos  # Uses English (default)")
        return
    
    if not check_script_exists():
        return
    
    # Get target directory from command line or use current directory
    target_dir = os.getcwd()  # Default to current directory
    
    # Parse arguments to find target directory (first non-flag argument that's not a language value)
    for i, arg in enumerate(sys.argv[1:], 1):
        if not arg.startswith('--'):
            # Check if this is a language value (following --lang)
            if i > 1 and sys.argv[i-1] == '--lang':
                continue  # Skip language values
            # This is the target directory
            target_dir = arg
            if not os.path.exists(target_dir):
                log(f"ERROR: Target directory '{target_dir}' does not exist!")
                return
            break
    
    log(f"Target directory: {target_dir}")
    
    # Find video files in the target directory
    files = sorted(glob.glob(os.path.join(target_dir, "*.mov")) + glob.glob(os.path.join(target_dir, "*.mp4")))
    
    if not files:
        log("No .mov or .mp4 files found in target directory.")
        return
    
    log(f"Batch processing started. Found {len(files)} files.")
    
    # Check for command line options
    args = []
    if "--no-refine" in sys.argv:
        args.append("--no-refine")
        log("Using --no-refine mode (faster processing)")
    if "--summary" in sys.argv:
        args.append("--summary")
        log("Using --summary mode (generating summaries)")
    if "--verbose" in sys.argv:
        args.append("--verbose")
        log("Using --verbose mode (detailed output)")
    if "--force" in sys.argv:
        args.append("--force")
        log("Using --force mode (overwrite existing files)")
    if "--lang" in sys.argv:
        lang_index = sys.argv.index("--lang")
        if lang_index + 1 < len(sys.argv) and not sys.argv[lang_index + 1].startswith("--"):
            lang_code = sys.argv[lang_index + 1]
            # Validate language code
            valid_languages = ['zh', 'en', 'ja', 'ko', 'fr', 'de', 'es', 'it', 'pt', 'ru']
            if lang_code in valid_languages:
                args.extend(["--lang", lang_code])
                log(f"Using language: {lang_code}")
            else:
                print(f"\nERROR: Invalid language code '{lang_code}'")
                print(f"Valid language options: {', '.join(valid_languages)}")
                print("\nPlease choose an option:")
                print("1. Enter a valid language code")
                print("2. Continue without --lang (use English default)")
                print("3. Exit")
                
                while True:
                    choice = input("\nEnter your choice (1/2/3): ").strip()
                    if choice == "1":
                        new_lang = input("Enter valid language code: ").strip().lower()
                        if new_lang in valid_languages:
                            args.extend(["--lang", new_lang])
                            log(f"Using corrected language: {new_lang}")
                            break
                        else:
                            print(f"Invalid language code '{new_lang}'. Please try again.")
                    elif choice == "2":
                        log("Continuing with English as default language.")
                        break
                    elif choice == "3":
                        log("Exiting batch processing.")
                        return
                    else:
                        print("Invalid choice. Please enter 1, 2, or 3.")
        else:
            log("WARNING: --lang specified but no language given. Using English as default.")

    successful = 0
    failed = 0
    
    for i, f in enumerate(files, 1):
        log(f"({i}/{len(files)}) Processing: {f}")
        start_time = time.time()
        
        try:
            # Build command with absolute paths
            cmd = ["python3", SCRIPT, f] + args
            
            # Language handling - English is default, but --lang can be specified
            if "--lang" not in args:
                log("Using English as default language. Use --lang to specify other languages.")
            
            # For verbose mode, show real-time output
            if "--verbose" in args:
                result = subprocess.run(
                    cmd,
                    timeout=12*3600  # 12h timeout in case of very large file
                )
            else:
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
                # Only show captured output in non-verbose mode
                if "--verbose" not in args:
                    if hasattr(result, 'stdout') and result.stdout:
                        log(f"STDOUT: {result.stdout}")
                    if hasattr(result, 'stderr') and result.stderr:
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
