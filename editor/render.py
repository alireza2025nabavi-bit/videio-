#!/usr/bin/env python3
"""Render the liquid-glass edit.

  python render.py                      # full film  -> ../AiWithMase_LiquidGlass_Edit.mp4
  python render.py --still 20 150 340   # single frames (PNG) into ./preview
  python render.py --sheet 44 58 1      # contact sheet of frames 44..58
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from lgk import scene as S  # noqa: E402

SRC_DEFAULT = os.path.join(ROOT, "AiWithMase final .mp4")
OUT_DEFAULT = os.path.join(ROOT, "AiWithMase_LiquidGlass_Edit.mp4")
SW, SH = 720, 1280


def ffmpeg_bin():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_source(src, cache):
    """Decode the whole clip once to raw RGB (BT.709, full range) and memory-map it."""
    need = S.NFRAMES * SW * SH * 3
    if not (os.path.exists(cache) and os.path.getsize(cache) == need):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-vf",
               "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=bicubic+accurate_rnd+full_chroma_int,format=rgb24",
               "-f", "rawvideo", cache]
        subprocess.run(cmd, check=True)
    return np.memmap(cache, dtype=np.uint8, mode="r", shape=(S.NFRAMES, SH, SW, 3))


def encoder(out, src, crf, preset):
    cmd = [ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{S.W}x{S.H}", "-r", str(S.FPS), "-i", "-",
           "-i", src,
           "-map", "0:v:0", "-map", "1:a:0",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p,"
                  "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high", "-level", "4.2",
           "-g", "48", "-bf", "2", "-maxrate", "24M", "-bufsize", "36M",
           "-x264-params", "aq-mode=3:aq-strength=0.9:deblock=-1,-1:colorprim=bt709:transfer=bt709:colormatrix=bt709",
           "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
           "-c:a", "copy", "-movflags", "+faststart", out]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--cache", default=os.path.join(HERE, ".cache", "source_rgb24.raw"))
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--still", type=int, nargs="*")
    ap.add_argument("--sheet", type=int, nargs=3, metavar=("FIRST", "LAST", "STEP"))
    ap.add_argument("--scale", type=float, default=0.5, help="contact sheet scale")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--preview-dir", default=os.path.join(HERE, "preview"))
    ap.add_argument("--no-ui", action="store_true")
    ap.add_argument("--range", type=int, nargs=2, metavar=("FIRST", "LAST"), help="render only part of the film")
    a = ap.parse_args()

    raw = load_source(a.src, a.cache)
    film = S.Film(raw)

    if a.still is not None or a.sheet is not None:
        os.makedirs(a.preview_dir, exist_ok=True)
        if a.still:
            for i in a.still:
                img = film.frame(i, ui=not a.no_ui)
                cv2.imwrite(os.path.join(a.preview_dir, f"f{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print("still", i)
        if a.sheet:
            first, last, step = a.sheet
            tiles = []
            for i in range(first, last + 1, step):
                img = film.frame(i, ui=not a.no_ui)
                img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(img, f"{i}  {i / S.FPS:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(img, f"{i}  {i / S.FPS:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                tiles.append(img)
            rows = []
            for r in range(0, len(tiles), a.cols):
                row = tiles[r:r + a.cols]
                while len(row) < a.cols:
                    row.append(np.zeros_like(tiles[0]))
                rows.append(np.hstack(row))
            path = os.path.join(a.preview_dir, f"sheet_{first:03d}_{last:03d}.jpg")
            cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
            print("sheet", path)
        return

    first, last = (a.range if a.range else (0, S.NFRAMES - 1))
    enc = encoder(a.out, a.src, a.crf, a.preset)
    t0 = time.time()
    for i in range(first, last + 1):
        enc.stdin.write(film.frame(i).tobytes())
        if i % 12 == 0:
            el = time.time() - t0
            done = i - first + 1
            print(f"frame {i:3d}/{last}  {done / max(el, 1e-6):4.1f} fps  eta {(last - i) * el / done:5.0f}s", flush=True)
    enc.stdin.close()
    enc.wait()
    print("wrote", a.out, f"({os.path.getsize(a.out) / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
