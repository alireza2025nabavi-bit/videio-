# Liquid-glass edit — PAMIR Lubricants 5W-30

This folder contains the complete, reproducible pipeline that turns
`AiWithMase final .mp4` into `AiWithMase_LiquidGlass_Edit.mp4`.

Everything is rendered in code (Python + OpenCV), frame by frame, at
1080 × 1920 / 24 fps. The original clip is never modified and its audio is
copied bit-for-bit, so the edit stays perfectly in sync with the music.

## Re-render

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python render.py                        # full film (about 6-8 minutes on 2 cores)
python render.py --still 20 150 340     # single frames -> preview/
python render.py --sheet 124 137 1      # contact sheet of a frame range -> preview/
```

`ffmpeg` is taken from `PATH`; if it is missing, the binary bundled with
`imageio-ffmpeg` is used.

## What the edit does

| Layer | Details |
|---|---|
| Liquid glass | Shapes are signed distance fields. Each one refracts the footage behind it through a lens-shaped bevel (with slight chromatic dispersion), frosts its interior, gets a light-aware specular rim, an inner shade and a soft contact shadow. Shapes fuse and split through a smooth minimum, like droplets. |
| The "island" | One piece of glass lives at the top of the frame for the whole film and morphs: title card → brand pill → spec bar that splits into three pills → brand pill → clear plaque behind the end lock-up. All moves are spring-driven and motion-blurred. |
| Feature pills | Lower-third bubbles that "pour out" a label: HIGH QUALITY, PREMIUM PERFORMANCE, HIGH DURABILITY. |
| Transitions | Zoom-through with real integrated motion blur and RGB split, impact punch + exposure flash on the big hits, and one signature liquid wave that floods the race shot and reveals the product. |
| Rhythm | The music is 128 BPM (first down-beat at 0.119 s). Reveals, splits, light sweeps and the small camera pulses are locked to that grid. |
| Weather | Continuous snowfall, composited in-camera: three depth layers of particles that fall, drift on the wind and sway, from 1 px motes to out-of-focus bokeh in front of the lens. Density is scored per shot and the flakes pick up light from what is behind them. |
| Petrol cars | The film is an engine-oil spot and the cars in shots 3, 8 and 9 are electric, so every electric car is replaced by a gold petrol sedan (BMW M5-style: quad chrome tips, boot lip, fuel flap, red LED tail-lights). A re-drawn still per key frame gives the car body; its difference against that frame gives the mask; the patch is then carried to the neighbouring frames with a chained optical-flow field measured on the original footage, feathered, exposure-matched and composited. The launch beat in shot 8 (where the near car leaves frame and its pose turns too far for a 2-D patch) is rebuilt from a single re-drawn plate pushed in slowly. Colour, lighting, reflections and timing stay the original. |
| Text repair | The tachometer dial in shot 7 misspells the brand (`PAMRI`). The shot is tracked frame by frame, the wrong letters are inpainted away and `PAMIR` is drawn back in the same typeface, size and softness, behind the needle. |
| Finish | Subtle filmic grade (cool shadows, warm highlights, richer gold), highlight bloom, vignette, fine film grain, fade-out. |

All on-screen copy is taken from the pack and the original end card
(`5W-30`, `PETROL ENGINE OIL`, `SAE`, `API SN`, `4 Liter`, the three feature
lines). Nothing was invented.

## Where to tweak

Everything creative sits in `lgk/scene.py`:

* `INTRO`, `BRAND`, `SPECS`, `FEATURES` – the texts and when the pills appear
* `ZOOM_CUTS`, `IMPACTS`, `PULSES` – transitions and beat accents
* `ST_CARD`, `ST_ISLAND`, `ST_SPEC`, `ST_PILL`, `ST_PLATE` – the glass looks
* `Island` – the choreography of the morphing glass
* `WIPE_F0`, `WIPE_F1` – the liquid wave
* `SNOW` – snowfall density per shot (0 = clear, 1 = heavy); the field itself
  (flake count, sizes, fall speed, wind) lives in `lgk/snow.py`

Supporting modules: `lgk/cars.py` (the petrol-car swap, plates in `assets/cars/`), `lgk/glass.py` (material renderer), `lgk/typo.py` (text with
tracking and real pair kerning, icons), `lgk/look.py` (camera, grade, bloom,
grain), `lgk/anim.py` (springs and easings), `lgk/snow.py` (the snowfall),
`lgk/fix.py` (tracking + text repair).

## Font

Montserrat (SIL Open Font License 1.1, see `fonts/OFL.txt`).
