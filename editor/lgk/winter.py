"""Winter ground: the night track is buried under snow.

The source clip is a wet, mirror-black race track.  This module lays a snow
surface over it, in the source frame, before the camera move and the grade, so
the snow moves with the shot and is graded like everything else.

How it is built
---------------
* the road is a polygon in source coordinates (the track only appears in the
  two race shots, so it is measured once per shot),
* the card is shaded by the *existing* light: the plate's luminance is blurred
  into a smooth shading map, so pools of lamp light, the dark under the cars and
  the glow of the PAMIR gantry all modulate the snow instead of a flat white,
* two noise fields - wide drifts and fine grain - break the surface up, and the
  bright painted lane lines stay faintly visible under a thinner layer,
* the car bodies are cut out of the coverage, so the tyres sit on the snow
  rather than behind it.
"""
import cv2
import numpy as np

H, W = 1280, 720

# road polygons in source pixels, per shot index
ROAD = {
    2: [(-140, 620), (360, 545), (860, 620), (860, 1440), (-140, 1440)],   # night straight
    7: [(-140, 665), (340, 600), (860, 665), (860, 1440), (-140, 1440)],   # launch plate
}

SNOW_RGB = np.array([0.870, 0.885, 0.920], np.float32)   # neutral white, a breath of blue


def _fbm(h, w, seed, base=10, octaves=5):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, norm, s = 1.0, 0.0, base
    for _ in range(octaves):
        gh, gw = max(2, h // s + 2), max(2, w // s + 2)
        g = rng.random((gh, gw)).astype(np.float32)
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += amp * g
        norm += amp
        amp *= 0.5
        s = max(2, int(s * 0.55))
    return acc / norm


class Ground:
    def __init__(self, cars, seed=4211):
        self.cars = cars
        self.seed = seed
        self._poly = {}
        self._tex = {}

    # ------------------------------------------------------------------ helpers
    def _poly_mask(self, shot):
        m = self._poly.get(shot)
        if m is None:
            m = np.zeros((H, W), np.float32)
            pts = np.array(ROAD[shot], np.int32)
            cv2.fillPoly(m, [pts], 1.0)
            m = cv2.GaussianBlur(m, (0, 0), 22.0)
            self._poly[shot] = m
        return m

    def _texture(self):
        if not self._tex:
            drift = _fbm(H, W, self.seed, base=90, octaves=4)
            mid = _fbm(H, W, self.seed + 1, base=26, octaves=4)
            fine = _fbm(H, W, self.seed + 2, base=4, octaves=3)
            self._tex = (drift, mid, fine)
        return self._tex

    # ------------------------------------------------------------------ main
    def apply(self, img, i, shot):
        """img: RGB uint8 (H, W, 3) -> snow-covered RGB uint8"""
        if shot not in ROAD:
            return img
        poly = self._poly_mask(shot)
        drift, mid, fine = self._texture()

        base = img.astype(np.float32) / 255.0
        luma = base @ np.array([0.299, 0.587, 0.114], np.float32)

        # --- shading: the snow keeps the light of the scene it lies in
        shade = cv2.GaussianBlur(luma, (0, 0), 12.0)
        ref = float(np.percentile(shade[poly > 0.5], 80)) + 1e-4
        shade_n = np.clip(shade / ref, 0.0, 2.0)
        light = np.clip(1.0 + 0.26 * (shade_n - 1.0), 0.76, 1.28)

        # --- surface: drifts (wide) + grain (fine)
        albedo = SNOW_RGB[None, None, :] * (light[..., None])
        albedo = albedo * (0.96 + 0.06 * (mid[..., None] - 0.5) * 2.0)
        albedo = albedo * (0.945 + 0.13 * (fine[..., None] - 0.5) * 2.0)

        # --- coverage: wavy drifts, thinner over the painted lane lines
        cov = poly * (0.96 + 0.22 * (drift - 0.5) * 2.0)
        bright = np.clip((luma - 0.55) / 0.30, 0.0, 1.0)
        cov = cov * (1.0 - 0.22 * bright)
        cov = np.clip(cov, 0.0, 1.0)

        # --- never paint over the cars
        car = self.cars.cover(i)
        if car is not None:
            core = cv2.dilate(car, np.ones((5, 5), np.uint8)).astype(np.float32) / 255.0
            core = cv2.GaussianBlur(core, (0, 0), 3.0)
            cov = cov * (1.0 - core)
            # a car sitting on snow throws a soft shadow into it, and that shadow
            # is what makes it sit on the ground instead of floating over it
            shadow = cv2.GaussianBlur(core, (0, 0), 26.0)
            albedo = albedo * (1.0 - 0.42 * np.clip(shadow * 2.2, 0.0, 1.0)[..., None])

        a = np.clip(cov, 0.0, 1.0)[..., None]
        out = base * (1.0 - a) + np.clip(albedo, 0.0, 1.0) * a
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
