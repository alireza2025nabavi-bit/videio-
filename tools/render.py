"""
PAMIR  "Black Edition"  —  frame-accurate re-grade / re-colour delivery.

  * decodes the source video frame by frame (no frame is skipped, dropped or
    duplicated - the encoder is fed exactly FRAME_COUNT frames),
  * paints every car body deep gloss black, keeping its highlights,
  * removes the Tesla emblems (tracked close-up badge + small distant marks),
  * adds liquid-glass title cards,
  * muxes the untouched original audio back in.

Usage
-----
  python tools/render.py                     # full render -> out/PAMIR_black_edition.mp4
  python tools/render.py --debug-masks       # dump per-shot mask contact sheets
  python tools/render.py --stills 0,128,312  # write processed stills for review
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import recolor as R           # noqa: E402
import glass as G             # noqa: E402
import titles as T            # noqa: E402

SRC = os.path.join(ROOT, "AiWithMase final .mp4")
OUT_DIR = os.path.join(ROOT, "out")
OUT = os.path.join(OUT_DIR, "PAMIR_black_edition.mp4")


# --------------------------------------------------------------------------- #
# shot table  (cut points verified on the source: 51 93 137 176 190 220 250 284)
# --------------------------------------------------------------------------- #
SHOTS = [
    dict(name="S1", f0=0,   f1=50,  paint=False),
    dict(name="S2", f0=51,  f1=92,  paint=True,  area=5000, y=(300, 790),
         exclude_rects=[("rect", 330, 0, 470, 700),      # pouring oil stream
                        ("rect", 0, 760, 720, 1280)]),   # oil in the canister
    dict(name="S3", f0=93,  f1=136, paint=True,  area=8000, y=(380, 830), clean=True),
    dict(name="S4", f0=137, f1=175, paint=False),
    dict(name="S5", f0=176, f1=189, paint=True,  area=4000, y=(0, 1280)),
    dict(name="S6", f0=190, f1=219, paint=False),
    dict(name="S7", f0=220, f1=249, paint=False),
    dict(name="S8", f0=250, f1=283, paint=True,  area=8000, y=(380, 830), clean=True),
    dict(name="S9", f0=284, f1=360, paint=True,  area=8000, y=(360, 900),
         # the gold product label of the canister must stay gold
         exclude_rects=[("rect", 0, 640, 320, 1280)]),
]


def shot_for(idx: int) -> dict:
    for s in SHOTS:
        if s["f0"] <= idx <= s["f1"]:
            return s
    return SHOTS[-1]


# --------------------------------------------------------------------------- #
def paint_frame(frame: np.ndarray, shot: dict,
                badge_pts: list[tuple[float, float]] | None = None) -> tuple[np.ndarray, dict]:
    stats = {"paint_px": 0, "badge_erased": 0}
    if not shot.get("paint"):
        return frame, stats

    region = None
    if shot.get("exclude_rects"):
        region = 255 - R.rect_mask(frame.shape, shot["exclude_rects"])
    y0, y1 = shot.get("y", (300, 1010))
    body = R.car_body(frame, area_min=shot.get("area", 8000),
                      y_min=y0, y_max=y1, region=region)
    stats["paint_px"] = int(body.sum())

    out = R.paint_black(frame, body)

    if shot.get("clean"):
        out = R.clean_pale_marks(out, body)

    if badge_pts:
        before = out.copy()
        out = R.erase_at(out, body, badge_pts, rx=13, ry=12, src_med=23)
        stats["badge_erased"] = int(np.abs(out.astype(np.int16) - before.astype(np.int16)).sum() > 0)
    return out, stats


# --------------------------------------------------------------------------- #
def build_cards() -> list[dict]:
    """Liquid-glass title cards + their on-screen life (frames, fade)."""
    cards = []
    for spec in T.TIMELINE:
        card = G.build_card(spec["w"], spec["h"],
                            eyebrow=spec.get("eyebrow", ""),
                            headline=spec.get("headline", ""),
                            sub=spec.get("sub", ""))
        cards.append({**spec, "card": card})
    return cards


def cards_for_frame(cards, idx):
    active = []
    for c in cards:
        a, b = c["f_in"], c["f_out"]
        if a - c["fade"] <= idx <= b + c["fade"]:
            # fade in / out + small rise
            if idx < a:
                k = (idx - (a - c["fade"])) / max(c["fade"], 1)
            elif idx > b:
                k = 1.0 - (idx - b) / max(c["fade"], 1)
            else:
                k = 1.0
            k = float(np.clip(k, 0, 1))
            ease = k * k * (3 - 2 * k)
            active.append((c, ease))
    return active


# --------------------------------------------------------------------------- #
def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise SystemExit("ffmpeg not found (pip install imageio-ffmpeg)")
        return exe


def decode_all(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    meta = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    meta["n_read"] = len(frames)
    return frames, meta


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-masks", action="store_true")
    ap.add_argument("--stills", type=str, default="")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    frames, meta = decode_all(SRC)
    print("source:", meta)
    n = len(frames)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- tracked badge (close-up shot S9) ---------------------------------
    # only the close-up window needs grayscale frames -> keeps RAM flat
    seed = T.BADGE_SEED
    g0 = max(0, min(s["f0"] for s in SHOTS if s["name"] == "S9") - 14)
    grays = {i: cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) for i in range(g0, n)}
    badge_path = R.track_badge(grays, seed["frame"], seed["pt"],
                               half=(15, 13), search=9, min_score=0.45)
    del grays
    print(f"badge tracked on {len(badge_path)} frames")

    if args.debug_masks:
        dbg = os.path.join(OUT_DIR, "debug")
        os.makedirs(dbg, exist_ok=True)
        for s in SHOTS:
            if not s.get("paint"):
                continue
            tiles = []
            for i in range(s["f0"], s["f1"] + 1, max(1, (s["f1"] - s["f0"]) // 5)):
                f = frames[i]
                region = None
                if s.get("exclude_rects"):
                    region = 255 - R.rect_mask(f.shape, s["exclude_rects"])
                y0, y1 = s.get("y", (300, 1010))
                body = R.car_body(f, area_min=s.get("area", 8000), y_min=y0, y_max=y1, region=region)
                vis = f.copy()
                vis[body > 0] = (0, 0, 255)
                vis = cv2.addWeighted(f, 0.45, vis, 0.55, 0)
                cv2.putText(vis, f"{s['name']} {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                tiles.append(cv2.resize(vis, (240, 427)))
            while len(tiles) % 6:
                tiles.append(np.zeros_like(tiles[0]))
            rows = [np.hstack(tiles[r * 6:(r + 1) * 6]) for r in range(len(tiles) // 6)]
            cv2.imwrite(os.path.join(dbg, f"mask_{s['name']}.png"), np.vstack(rows))
            print("wrote", os.path.join(dbg, f"mask_{s['name']}.png"))
        return

    cards = build_cards()

    if args.stills:
        idxs = [int(x) for x in args.stills.split(",")]
        for i in idxs:
            s = shot_for(i)
            bp = [badge_path[i]] if (s["name"] == "S9" and i in badge_path) else None
            out, st = paint_frame(frames[i], s, bp)
            for c, k in cards_for_frame(cards, i):
                out = G.composite(out, c["card"], c["x"], c["y"] + int((1 - k) * 8),
                                  opacity=k)
            cv2.imwrite(os.path.join(OUT_DIR, f"still_{i:03d}.png"), out)
            print("still", i, s["name"], st)
        return

    if args.dry_run:
        return

    # ---- streaming encode --------------------------------------------------
    fh, fw = meta["h"], meta["w"]
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{fw}x{fh}",
           "-r", f"{meta['fps']:.6f}", "-i", "-",
           "-i", SRC,
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
           "-x264-params", "keyint=48:min-keyint=24:ref=4:bframes=3",
           "-c:a", "copy", "-movflags", "+faststart", "-shortest", OUT]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    log = []
    for i, f in enumerate(frames):
        s = shot_for(i)
        bp = [badge_path[i]] if (s["name"] == "S9" and i in badge_path) else None
        out, st = paint_frame(f, s, bp)
        for c, k in cards_for_frame(cards, i):
            out = G.composite(out, c["card"], c["x"], c["y"] + int((1 - k) * 8), opacity=k)
        proc.stdin.write(out.tobytes())
        frames[i] = None                      # free as we go
        log.append({"f": i, "shot": s["name"], **st})
        if i % 60 == 0:
            print("encoded", i, "/", n, end="\r", flush=True)
    proc.stdin.close()
    proc.wait()
    print("\nffmpeg exit", proc.returncode)
    with open(os.path.join(OUT_DIR, "frame_log.json"), "w") as f:
        json.dump(log, f)
    if proc.returncode != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
