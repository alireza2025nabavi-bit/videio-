"""
Frame-accurate car recolour engine for the PAMIR "Black Edition" edit.

Everything here works on single BGR frames so the caller can stream
decode -> process -> encode without ever dropping or reordering a frame.
"""
from __future__ import annotations

import cv2
import numpy as np


# ----------------------------------------------------------------------------- #
# 1. car body detection
# ----------------------------------------------------------------------------- #
def gold_mask(bgr: np.ndarray,
              h_lo: int = 7, h_hi: int = 35,
              s_min: int = 85, v_min: int = 55) -> np.ndarray:
    """Raw HSV mask of the gold/champagne paint."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = ((h > h_lo) & (h < h_hi) & (s > s_min) & (v > v_min)).astype(np.uint8)
    return m


def car_body(bgr: np.ndarray,
             area_min: int = 8000,
             y_min: int = 330, y_max: int = 1010,
             x_min: int = 0,
             region: np.ndarray | None = None,
             fill: int = 13) -> np.ndarray:
    """
    Paint mask = large gold blobs inside the expected band, holes filled,
    optionally restricted by a region mask (per-shot ROI).
    """
    m = gold_mask(bgr)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < area_min:
            continue
        cx, cy = cent[i]
        if not (y_min < cy < y_max):
            continue
        if cx <= x_min:
            continue
        keep[lab == i] = 1

    if region is not None:
        keep = cv2.bitwise_and(keep, (region > 0).astype(np.uint8))

    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # fill the small holes inside the silhouette (specular highlights, trim),
    # but never swallow big holes (windows, tail lights, background).
    cont, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(keep)
    cv2.drawContours(filled, cont, -1, 1, -1)
    near = cv2.dilate(keep, np.ones((fill, fill), np.uint8))
    body = cv2.bitwise_and(filled, near)
    body = cv2.bitwise_or(body, keep)
    if region is not None:
        body = cv2.bitwise_and(body, (region > 0).astype(np.uint8))
    return body


# ----------------------------------------------------------------------------- #
# 2. gold -> gloss black paint
# ----------------------------------------------------------------------------- #
def paint_black(bgr: np.ndarray, body: np.ndarray,
                gamma: float = 2.75,
                desat: float = 0.94,
                sat_norm: float = 0.40,
                feather: float = 2.0,
                grow: int = 2,
                cool: float = 2.0) -> np.ndarray:
    """
    Keep the luminance structure of the paint (so highlights, reflections and
    the glossy black look survive) while removing the gold chroma and
    dropping the mid-tones to a deep black.
    """
    if grow:
        body = cv2.dilate(body, np.ones((grow, grow), np.uint8))
    alpha = cv2.GaussianBlur(body.astype(np.float32), (0, 0), feather)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    # pixels that are already neutral (glass, chrome, specular) are protected
    w = np.clip(alpha * np.clip(sat / sat_norm, 0.0, 1.0), 0.0, 1.0)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    L_t = 255.0 * np.power(np.clip(L / 255.0, 0.0, 1.0), gamma)
    A_t = 128.0 + (A - 128.0) * (1.0 - desat * w)
    B_t = 128.0 + (B - 128.0) * (1.0 - desat * w) - cool * w

    L = L * (1.0 - w) + L_t * w
    out = cv2.merge([L, A_t, B_t]).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


# ----------------------------------------------------------------------------- #
# 3. emblem / badge removal
# ----------------------------------------------------------------------------- #
def _med_fill(img: np.ndarray, mask: np.ndarray, k: int, feather: float) -> np.ndarray:
    w = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), feather)
    w = np.clip(w, 0, 1)[..., None]
    src = cv2.medianBlur(img, k).astype(np.float32)
    return (img.astype(np.float32) * (1 - w) + src * w).clip(0, 255).astype(np.uint8)


def clean_pale_marks(img: np.ndarray, body: np.ndarray,
                     resp_thr: int = 18,
                     area_max: int = 900,
                     bg_med: int = 31,
                     src_med: int = 17,
                     dilate: int = 7,
                     feather: float = 2.5) -> np.ndarray:
    """
    Removes small pale marks sitting on the (now very dark) paint - this is
    what erases the tiny distant emblems without touching large reflections.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(g, bg_med)
    resp = g.astype(np.int16) - bg.astype(np.int16)
    core = cv2.erode(body, np.ones((5, 5), np.uint8))
    cand = ((resp > resp_thr) & (core > 0)).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    keep = np.zeros_like(cand)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= area_max:
            keep[lab == i] = 255
    if keep.sum() == 0:
        return img
    keep = cv2.dilate(keep, np.ones((dilate, dilate), np.uint8))
    keep = cv2.bitwise_and(keep, core * 255)
    return _med_fill(img, keep, src_med, feather)


def erase_at(img: np.ndarray, body: np.ndarray, pts: list[tuple[float, float]],
             rx: float = 13.0, ry: float = 11.0, feather: float = 2.0,
             src_med: int = 21) -> np.ndarray:
    """
    Erases an emblem at known coordinates (used for the big close-up badge).
    The fill is clamped to the paint mask so nothing else is ever touched.
    """
    if not pts:
        return img
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for x, y in pts:
        if x < -40 or y < -40 or x > w + 40 or y > h + 40:
            continue
        cv2.ellipse(m, (int(round(x)), int(round(y))), (int(rx), int(ry)),
                    0, 0, 360, 255, -1)
    m = cv2.bitwise_and(m, cv2.erode(body, np.ones((3, 3), np.uint8)) * 255)
    if m.sum() == 0:
        return img
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return _med_fill(img, m, src_med, feather)


# ----------------------------------------------------------------------------- #
# 4. badge tracking (sub-pixel-ish patch tracker, verified per shot)
# ----------------------------------------------------------------------------- #
def track_badge(grays: dict[int, np.ndarray], seed: int, pt: tuple[float, float],
                half: tuple[int, int] = (15, 13),
                search: int = 9,
                min_score: float = 0.45) -> dict[int, tuple[float, float]]:
    """
    Adaptive template tracker.  `grays` maps frame index -> grayscale frame.
    Returns frame -> (x, y) for the frames it could follow (nearest hold).
    """
    idxs = sorted(grays)
    lo, hi = idxs[0], idxs[-1]
    hw, hh = half
    path: dict[int, tuple[float, float]] = {seed: (float(pt[0]), float(pt[1]))}

    def patch(g, p):
        x, y = int(round(p[0])), int(round(p[1]))
        x0, y0 = max(0, x - hw), max(0, y - hh)
        return g[y0:y0 + 2 * hh + 1, x0:x0 + 2 * hw + 1]

    for step in (1, -1):
        cur = (float(pt[0]), float(pt[1]))
        tpl = patch(grays[seed], cur).astype(np.float32)
        i = seed
        while True:
            nxt = i + step
            if nxt < lo or nxt > hi:
                break
            g = grays[nxt]
            x0 = max(0, int(round(cur[0])) - hw - search)
            y0 = max(0, int(round(cur[1])) - hh - search)
            x1 = min(g.shape[1], int(round(cur[0])) + hw + search + 1)
            y1 = min(g.shape[0], int(round(cur[1])) + hh + search + 1)
            roi = g[y0:y1, x0:x1]
            if roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]:
                break
            r = cv2.matchTemplate(roi.astype(np.float32), tpl.astype(np.float32),
                                  cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(r)
            if score >= min_score:
                cur_add = (x0 + loc[0] + hw, y0 + loc[1] + hh)
                cur_add = (0.5 * (cur_add[0] + cur[0] + (cur_add[0] - cur[0]) * 0.0),
                           0.5 * (cur_add[1] + cur[1]))
                cur = cur_add
                new_tpl = patch(g, cur).astype(np.float32)
                if new_tpl.shape == tpl.shape:
                    tpl = 0.6 * tpl + 0.4 * new_tpl
            path[nxt] = cur
            i = nxt
    return path


# ----------------------------------------------------------------------------- #
# 5. region helpers
# ----------------------------------------------------------------------------- #
def poly_mask(shape, polys) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.array(p, np.int32)], 255)
    return m


def rect_mask(shape, rects) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    for x0_flag, x0, y0, x1, y1 in rects:
        if x0_flag == "rect":
            m[max(0, y0):y1, max(0, x0):x1] = 255
    return m
