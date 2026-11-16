"""Run one project example for the GUI worker process."""

import argparse
import os
import runpy
import sys
from pathlib import Path

# Paths are relative to their chapter directory, matching GUI.py's lookup.
CAMERA_CASES = {
    "OpenCV Örnekleri/canliGoruntu.py",
    "Yüz işlemleri/live.py",
    "Renk takibi/program.py",
    "Göz ile fare kontrolü/app.py",
}


def run_worker(script, preview_dir, face_image, mode, script_args):
    del face_image, mode  # Reserved for the batch verifier interface.
    script = Path(script).resolve()
    Path(preview_dir).resolve().mkdir(parents=True, exist_ok=True)
    if not script.is_file():
        raise FileNotFoundError(f"Örnek dosyası bulunamadı: {script}")
    os.chdir(script.parent)
    sys.argv = [str(script), *script_args]
    runpy.run_path(str(script), run_name="__main__")


def main():
    parser = argparse.ArgumentParser(description="KODLAB örnek çalıştırıcısı")
    parser.add_argument("--worker", nargs=4, metavar=("SCRIPT", "PREVIEW", "FACE", "MODE"))
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.worker:
        parser.error("Bu yardımcı program GUI için --worker ile çağrılmalıdır.")
    run_worker(*args.worker, args.script_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
