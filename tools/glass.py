"""
Liquid-glass (glassmorphism) title cards.

The card is rendered once with PIL at 2x and down-sampled for clean edges;
at composite time the *video* behind the card is blurred so that the panel
really behaves like frosted glass instead of a flat translucent rectangle.
"""
from __future__ import annotations

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

import arabic_reshaper
from bidi.algorithm import get_display

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "..", "assets", "fonts")


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def fa(text: str) -> str:
    """Shape + reorder Persian/Arabic text so Pillow draws it correctly."""
    return get_display(arabic_reshaper.reshape(text))


# --------------------------------------------------------------------------- #
def _rounded(size, radius):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1],
                                        radius=radius, fill=255)
    return m


def build_card(width: int, height: int,
               eyebrow: str = "", headline: str = "",
               sub: str = "", rtl: bool = True,
               radius: int = 26,
               pad: int = 26,
               accent: tuple[int, int, int] = (255, 214, 130),
               scale: int = 2):
    """
    Returns dict with float32 arrays:
      rgb  (h,w,3)  card colour
      a    (h,w)    card opacity
      sh   (h,w)    soft drop-shadow opacity (same size, offset applied later)
    """
    W, H = width * scale, height * scale
    rgb = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)

    mask = np.array(_rounded((W, H), radius * scale), np.float32) / 255.0

    # --- body: vertical white gradient (brighter at the top = light hitting glass)
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    grad = 0.34 - 0.20 * yy                      # 0.34 -> 0.14
    body_a = (grad * mask)

    # --- edge light: thin bright rim, strongest on the top edge
    inner = np.array(_rounded((W, H), radius * scale), np.float32) / 255.0
    edge_img = Image.fromarray((inner * 255).astype(np.uint8))
    edge = np.array(edge_img.filter(ImageFilter.FIND_EDGES), np.float32) / 255.0
    edge = cv2.GaussianBlur(edge, (0, 0), 0.9 * scale)
    edge = np.clip(edge * 2.6, 0, 1) * np.clip(1.25 - yy, 0, 1)
    edge *= mask

    # --- specular sweep across the glass
    xs = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    sweep = np.exp(-((xs - 0.22 + yy * 0.28) ** 2) / (2 * 0.10 ** 2)) * 0.16
    sweep *= mask

    a_body = np.clip(body_a + edge * 0.55 + sweep, 0, 0.92)
    rgb += np.array([248, 250, 255], np.float32) * (a_body[..., None])

    # accent hairline on the left inner edge (brand warmth)
    gap = np.zeros((H, W), np.float32)
    gap[:, int(0.008 * W):int(0.014 * W)] = 1
    gap *= mask
    rgb += np.array(accent, np.float32) * (gap[..., None] * 0.30)
    a_body = np.clip(a_body + gap * 0.25, 0, 0.95)

    # --- text -------------------------------------------------------------
    img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    a_img = Image.fromarray(np.clip(a_body * 255, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    da = ImageDraw.Draw(a_img)

    x_right = W - pad * scale
    x_left = pad * scale
    cursor_y = pad * scale

    if eyebrow:
        f = _font("Vazirmatn-Medium.ttf", int(13 * scale))
        txt = eyebrow.upper()
        bb = d.textbbox((0, 0), txt, font=f)
        tw = bb[2] - bb[0]
        pos = (x_right - tw, cursor_y)
        d.text(pos, txt, font=f, fill=(255, 236, 205))
        da.text(pos, txt, font=f, fill=255)
        cursor_y += int(21 * scale)

    if headline:
        fs = int(31 * scale)
        f = _font("Vazirmatn-Bold.ttf", fs)
        txt = fa(headline) if rtl else headline
        bb = d.textbbox((0, 0), txt, font=f)
        tw = bb[2] - bb[0]
        pos = (x_right - tw, cursor_y - int(3 * scale))
        d.text(pos, txt, font=f, fill=(255, 255, 255))
        da.text(pos, txt, font=f, fill=255)
        cursor_y += int(fs * 1.06)

    if sub:
        f = _font("Vazirmatn-Regular.ttf", int(14 * scale))
        txt = sub
        bb = d.textbbox((0, 0), txt, font=f)
        tw = bb[2] - bb[0]
        pos = (x_right - tw, cursor_y + int(5 * scale))
        d.text(pos, txt, font=f, fill=(236, 240, 250))
        da.text(pos, txt, font=f, fill=235)

    rgb = np.array(img, np.float32) / 255.0
    alpha = np.array(a_img, np.float32) / 255.0

    # --- drop shadow ------------------------------------------------------
    sh = np.array(_rounded((W, H), radius * scale), np.float32) / 255.0
    sh = cv2.GaussianBlur(sh, (0, 0), 9 * scale) * 0.55

    # down-sample (anti alias)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)
    sh = cv2.resize(sh, (width, height), interpolation=cv2.INTER_AREA)
    return {"rgb": rgb, "a": alpha, "sh": sh}


# --------------------------------------------------------------------------- #
def composite(frame: np.ndarray, card: dict, x: int, y: int,
              opacity: float = 1.0, frost: float = 0.85, blur: float = 11.0,
              shadow_dy: int = 10) -> np.ndarray:
    """Paste a glass card on the frame with real backdrop frosting."""
    if opacity <= 0.002:
        return frame
    h, w = card["a"].shape
    H, W = frame.shape[:2]
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + w, y0 + h
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return frame

    out = frame.astype(np.float32)

    # shadow first
    sh = (card["sh"] * opacity)[..., None]
    sy0, sy1 = y0 + shadow_dy, y1 + shadow_dy
    if sy1 <= H:
        out[sy0:sy1, x0:x1] *= (1.0 - sh * 0.9)

    region = out[y0:y1, x0:x1]
    back = cv2.GaussianBlur(region, (0, 0), blur)
    back = back * frost + region * (1.0 - frost)
    back = np.clip(back * 0.97 + 9.0, 0, 255)

    a = (card["a"] * opacity)[..., None]
    comp = back * (1.0 - a) + card["rgb"] * 255.0 * a
    out[y0:y1, x0:x1] = comp
    return np.clip(out, 0, 255).astype(np.uint8)
