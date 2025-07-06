import os
import subprocess
import sys


def main():
    """Downscale large MOV files to 480p resolution."""
    if len(sys.argv) < 2:
        print("Usage: python batch_downscale_480p.py /path/to/output_folder")
        sys.exit(1)

    output_folder = sys.argv[1]
    if not os.path.isdir(output_folder):
        print(f"Output folder does not exist: {output_folder}")
        sys.exit(1)

    # Size threshold: 3GB in bytes
    threshold = 3 * 1024 * 1024 * 1024

    for filename in os.listdir("."):
        if filename.lower().endswith(".mov"):
            filepath = os.path.abspath(filename)
            if os.path.getsize(filepath) > threshold:
                base, ext = os.path.splitext(filename)
                out_name = f"{base}{ext}"
                out_path = os.path.join(output_folder, out_name)
                if os.path.exists(out_path):
                    print(f"Skipping (already exists): {out_path}")
                    continue
                print(f"Converting: {filename} → {out_path}")
                cmd = [
                    "ffmpeg",
                    "-i", filepath,
                    "-vf", "scale=640:480",
                    "-c:v", "libx264",
                    "-crf", "23",
                    "-c:a", "aac",
                    out_path
                ]
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
