# Clean up images unreadable by PIL from a directory. Use this with GNU parallel
# to filter tons.
from pathlib import Path
import sys


def clean_img_dir(dir):
    for path in os.listdir(dir):
        try:
            PIL.Image.open(f"{dir}/{path}").copy()
        except (PIL.UnidentifiedImageError, OSError):
            print(f"Deleting {path}")
            os.remove(f"{dir}/{path}")


for dir in sys.argv[1:]:
    dir = Path(dir)
    if dir.is_dir():
        print(f"Cleaning {dir}")
        clean_img_dir(dir)
        print(f"Done cleaning {dir}")
