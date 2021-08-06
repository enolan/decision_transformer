# Take a directory from a RipMe rip of a subreddit and process it for later use.

# NB for some reason it doesn't download the full set of posts.
# https://github.com/RipMeApp/ripme/issues/1898
# Maybe a ratelimiting thing? They could lie about how many posts there are if
# they detect you dumping.

import imageio_ffmpeg
from pathlib import Path
from PIL import Image
import random
from subprocess import CalledProcessError
import sys
from tqdm import tqdm

top_dir = Path(sys.argv[1])

outnum = 0

stills_out_dir = top_dir.parent / f"{top_dir.name}_stills"
video_stills_out_dir = top_dir.parent / f"{top_dir.name}_video_stills"

for dir in [stills_out_dir, video_stills_out_dir]:
    dir.mkdir(exist_ok=True)
    print(f"Using output directory {dir}")

provenance = []


def process_dir(dir, depth):
    global outnum
    paths = list(dir.iterdir())  # Make a list so we can get an ETA with tqdm
    for p in tqdm(paths, position=depth, leave=depth == 0):
        if p.is_file():
            if p.suffix == ".jpg" or p.suffix == ".jpeg" or p.suffix == ".png":
                target = (stills_out_dir / str(outnum)).with_suffix(p.suffix)
                target.symlink_to(p)
                provenance.append((p, target))
            elif p.suffix == ".mp4" or p.suffix == ".gif":
                # For videos we save a random still image. Could do more than
                # one? IDK what ratio would be good. Maybe 1 per 5 seconds?
                try:
                    frames = imageio_ffmpeg.count_frames_and_secs(p)[0]
                    frame_to_save = random.randint(0, frames - 1)
                    gen = imageio_ffmpeg.read_frames(p)
                    metadata = gen.__next__()
                    for n in range(frame_to_save - 1):  # first output is metadata dict
                        gen.__next__()
                    frame_buffer = gen.__next__()
                    frame_pil = Image.frombytes(
                        mode="RGB", size=metadata["source_size"], data=frame_buffer
                    )
                    target = (video_stills_out_dir / str(outnum)).with_suffix(".jpg")
                    frame_pil.save(target, quality=90)
                    provenance.append((p, target))
                except (RuntimeError, StopIteration) as e:
                    # Sometimes it fails when calling gen.__next__(), I'm not
                    # sure why. Maybe the frame counts reported are inaccurate?
                    print(f"Couldn't process {p} with ffmpeg: {e}")
            else:
                print(f"Unknown suffix {p.suffix}")
            outnum = outnum + 1
        else:
            process_dir(p, depth + 1)


process_dir(top_dir, 0)

with open(
    top_dir.parent / f"{top_dir.name}_provenance",
    mode="w",
) as f:
    for source, target in provenance:
        f.write(f"{source} -> {target}\n")
