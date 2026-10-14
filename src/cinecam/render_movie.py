"""
Render all the frames of a cinecam camera path with shamrock.

    ./cinecam render campath.json --out frames/ [--frames 0:300] [--scale 0.5] [--mp4 movie.mp4]

Frames are written as frame_%05d.png. Use --frames to split a long movie between
several jobs / GPUs (--sycl), and --skip-existing to resume an interrupted render.
"""

import argparse
import os
import subprocess
import sys
import time

from .data.sinktracks import SinkTracks
from .path.campath import CameraPath
from .render.backend import FrameRenderer, SimScene, init_shamrock
from .render.compose import save_png
from .render.stars import star_exposure


def main():
    ap = argparse.ArgumentParser(prog="cinecam render", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="camera path JSON written by the editor")
    ap.add_argument("--out", default=None, help="output directory (default: <path>_frames)")
    ap.add_argument("--frames", default=None, help="frame range a:b (python slice, b excluded)")
    ap.add_argument("--scale", type=float, default=1.0, help="resolution scale, e.g. 0.5 for a draft")
    ap.add_argument("--sycl", default="0:0", help="shamrock sycl config")
    ap.add_argument("--mp4", default=None, help="encode the frames into this movie with ffmpeg at the end")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality for --mp4 (lower is better)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--format", default=None, help="dump format of the fast sink reader (default: the path's)")
    args = ap.parse_args()

    path = CameraPath.load(args.path)
    if not path.keyframes:
        sys.exit("ERROR: the camera path has no keyframe")
    if not path.dumps:
        sys.exit("ERROR: the camera path has no simulation dump")
    out = args.out or os.path.splitext(args.path)[0] + "_frames"
    os.makedirs(out, exist_ok=True)

    nx = int(round(path.resolution[0] * args.scale / 2)) * 2  # even sizes for yuv420p
    ny = int(round(path.resolution[1] * args.scale / 2)) * 2
    n = path.n_frames
    a, b = 0, n
    if args.frames:
        sa, sb = args.frames.split(":")
        a = int(sa) if sa else 0
        b = min(int(sb), n) if sb else n
    frames = range(a, b)

    init_shamrock(args.sycl)
    scene = SimScene()
    renderer = FrameRenderer(scene)
    # sink trajectories: cameras following a sink, sinks drawn between dumps
    t0 = time.perf_counter()
    tracks = SinkTracks.build(path.dumps, fmt=args.format or path.format)
    path.set_tracks(tracks)
    renderer.tracks = tracks
    print(f"sink tracks: {tracks.nsink} sinks in {tracks.ndump} dumps ({time.perf_counter() - t0:.1f} s)", flush=True)
    print(f"rendering frames {a}..{b - 1} of {n} at {nx}x{ny} into {out}", flush=True)

    t_start = time.perf_counter()
    for k, i in enumerate(frames):
        fname = os.path.join(out, f"frame_{i:05d}.png")
        if args.skip_existing and os.path.exists(fname):
            continue
        t0 = time.perf_counter()
        st = path.evaluate(path.frame_time(i))
        if scene.load(path.dump_path(st)):
            print(f"loaded {scene.path} (t = {scene.time_myr:.3f} Myr, {scene.load_time:.1f} s)", flush=True)
        timings = {}
        exposure = star_exposure(path, path.frame_time(i), renderer.starlight, nx, ny)
        img = renderer.render(st, nx, ny, path.look, timings, exposure)
        save_png(img, fname)
        dt = time.perf_counter() - t0
        el = time.perf_counter() - t_start
        eta = el / (k + 1) * (len(frames) - k - 1)
        print(
            f"PROGRESS {k + 1}/{len(frames)} frame {i} in {dt:.1f} s "
            f"(gas {timings.get('gas', 0):.1f} s, stars {timings.get('stars', 0):.1f} s), ETA {eta / 60:.1f} min",
            flush=True,
        )

    if args.mp4:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", f"{path.fps:g}",
            "-start_number", str(a),
            "-i", os.path.join(out, "frame_%05d.png"),
            "-frames:v", str(len(frames)),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
            args.mp4,
        ]  # fmt: skip
        print("ffmpeg: " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    print(f"DONE {len(frames)} frames in {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
