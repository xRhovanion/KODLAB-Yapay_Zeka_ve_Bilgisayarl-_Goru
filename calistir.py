"""Shared project discovery helpers used by the GUI and the console launcher."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHAPTER_PREFIXES = ("2.", "3.", "4.", "6.", "7.")
HELPER_FILES = {"vectors.py", "fuzzyPy.py", "TFmodel.py", "svm_demo.py"}


def project_scripts():
    """Return the runnable book examples in a stable, human-friendly order."""
    scripts = []
    for chapter in sorted(ROOT.iterdir(), key=lambda p: p.name.casefold()):
        if not chapter.is_dir() or not chapter.name.startswith(CHAPTER_PREFIXES):
            continue
        scripts.extend(
            path
            for path in chapter.rglob("*.py")
            if path.name not in HELPER_FILES and path.name != "__init__.py"
        )
    return sorted(scripts, key=lambda p: str(p.relative_to(ROOT)).casefold())


if __name__ == "__main__":
    for script in project_scripts():
        print(script.relative_to(ROOT))
