"""Copy + timing for the liquid-glass title cards (24 fps timebase)."""

CARD_W = 608
CARD_H = 140
CARD_X = 56

TIMELINE = [
    # ---- hook: the product in the warehouse ---------------------------------
    dict(f_in=8,  f_out=44,  fade=8, x=CARD_X, y=92, w=CARD_W, h=CARD_H,
         eyebrow="PAMIR LUBRICANTS",
         headline="روغن موتور تمام‌سینتتیک",
         sub="SAE 5W-30  ·  API SN  ·  4L"),

    # ---- drag race ----------------------------------------------------------
    dict(f_in=99, f_out=129, fade=8, x=CARD_X, y=92, w=CARD_W, h=CARD_H,
         eyebrow="POWER IN EVERY DROP",
         headline="قدرت در هر قطره",
         sub="FULLY SYNTHETIC  ·  5W-30"),

    # ---- engine internals ---------------------------------------------------
    dict(f_in=196, f_out=214, fade=7, x=CARD_X, y=92, w=CARD_W, h=CARD_H,
         eyebrow="THERMAL PROTECTION",
         headline="محافظت در دمای بالا",
         sub="OIL FILM STABLE AT 150°C"),

    # ---- second race --------------------------------------------------------
    dict(f_in=256, f_out=278, fade=7, x=CARD_X, y=92, w=CARD_W, h=CARD_H,
         eyebrow="UNMATCHED ACCELERATION",
         headline="شتاب بی‌مانند",
         sub="PAMIR PERFORMANCE SERIES"),

    # ---- final hero shot ----------------------------------------------------
    dict(f_in=300, f_out=352, fade=8, x=CARD_X, y=92, w=CARD_W, h=CARD_H,
         eyebrow="PAMIR  BLACK  EDITION",
         headline="پامیر بلک ادیشن",
         sub="5W-30  ·  API SN  ·  FULLY SYNTHETIC"),
]

# seed for the tracked close-up emblem removal (frame index + x, y)
BADGE_SEED = {"frame": 312, "pt": (665.0, 599.0)}
