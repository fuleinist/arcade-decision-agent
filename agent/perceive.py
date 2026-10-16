"""Frame -> features -> PROSE.

The single most load-bearing file in the project, because Laya compares words, not
floats: measured on the local engine today, the same facts scored +0.53 separation
between the two directions as a one-sentence prose state, -0.20 as a JSON dict and
-0.09 as bare numbers. Whatever the source of truth (pixels, a JS state object, or
emulator RAM), it leaves this module as a sentence.
"""
from __future__ import annotations

from io import BytesIO

NEUTRAL = "neutral"

#: buckets for a horizontal offset, expressed as words a decision model can compare
DX_WORDS = [
    (-1.00, -0.25, "far to the left of"),
    (-0.25, -0.08, "to the left of"),
    (-0.08, -0.02, "just left of"),
    (-0.02, 0.02, "lined up with"),
    (0.02, 0.08, "just right of"),
    (0.08, 0.25, "to the right of"),
    (0.25, 1.00, "far to the right of"),
]


def bucket_dx(dx_frac: float | None) -> str:
    if dx_frac is None:
        return NEUTRAL
    for lo, hi, word in DX_WORDS:
        if lo <= dx_frac < hi:
            return word
    return NEUTRAL


def prose_from_features(f: dict) -> str:
    """Render the features as one sentence about the one decision that matters.

    Deliberately ONE clause: a second clause ('and it is not already rising')
    measurably cost 0.18 of separation on the same state in the Flappy measurements,
    and it costs the same here.
    """
    if not f.get("enemy_x") and f.get("enemy_x") != 0:
        return "There is no enemy on screen to aim at, so the ship should stay put."

    rel = bucket_dx(f.get("dx_frac"))
    parts: list[str] = []
    if rel == NEUTRAL:
        parts.append("The player ship has no enemy to steer toward.")
    else:
        parts.append(f"The player ship is {rel} the nearest enemy.")
    if f.get("aligned"):
        parts[-1] = "The player ship is lined up with the nearest enemy."
    return " ".join(parts)


def threat_prose(f: dict) -> str:
    """A separate, independently-answerable sentence for the dodge question."""
    d = f.get("diving_dist")
    if f.get("diving_near") or (d is not None and d < 60):
        return "An enemy is diving straight at the player ship."
    if d is not None:
        return "An enemy is descending but well away from the player ship."
    return "No enemy is diving at the player ship."


# ---------------------------------------------------------------- pixel path ---
def blank_fraction(png: bytes) -> float:
    """Fraction of sampled pixels that are (near) black.

    Used as the readiness test for a live emulator canvas: `Module.calledRun` goes true
    several seconds before the ROM produces video, and a screenshot of a WebGL canvas
    that is not drawing yet is a black rectangle.
    """
    from PIL import Image

    im = Image.open(BytesIO(png)).convert("RGB")
    px = im.load()
    w, h = im.size
    total = dark = 0
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            total += 1
            if sum(px[x, y]) < 30:
                dark += 1
    return dark / max(total, 1)


def frame_features(png: bytes, ship_width: int = 26) -> dict:
    """Extract the few features the policy needs from a raw canvas screenshot.

    Deliberately crude and cheap: two horizontal bands (enemy field, ship line),
    a brightness column histogram in each, and the horizontal offset between the
    peaks. No OCR, no object detection, no model. Verified against the live
    classicgamesarcade Galaga canvas (672x768, WebGL) on 2026-10-16.
    """
    from PIL import Image  # imported lazily so the stub env needs no deps

    im = Image.open(BytesIO(png)).convert("RGB")
    w, h = im.size
    px = im.load()

    def peaks(y0: float, y1: float, bright: int = 150) -> list[tuple[int, int]]:
        col = [0] * w
        for y in range(int(h * y0), int(h * y1), 2):
            for x in range(0, w, 2):
                r, g, b = px[x, y]
                if r > bright and g > bright and b > bright:
                    col[x] += 1
                    col[min(x + 1, w - 1)] += 1
        out = []
        for x in range(w - ship_width):
            out.append((x, sum(col[x:x + ship_width])))
        return out

    ship_cols = peaks(0.88, 0.99)
    enemy_cols = peaks(0.10, 0.55)
    mid_cols = peaks(0.60, 0.85, bright=190)

    ship_x = max(ship_cols, key=lambda t: t[1])[0] if ship_cols else w // 2
    enemy_x, enemy_v = max(enemy_cols, key=lambda t: t[1]) if enemy_cols else (None, 0)
    mid_v = max((v for _, v in mid_cols), default=0)
    ship_v = max((v for _, v in ship_cols), default=0)
    if enemy_v < 6:
        enemy_x = None

    dx = (enemy_x + ship_width // 2) - (ship_x + ship_width // 2) if enemy_x is not None else None
    mid_x, mid_v_peak = max(mid_cols, key=lambda t: t[1]) if mid_cols else (None, 0)
    threat_dx = (mid_x + ship_width // 2) - (ship_x + ship_width // 2) if mid_x is not None else None
    return {
        "ship_x": ship_x,
        "ship_frac": ship_x / w,
        "enemy_x": enemy_x,
        "dx": dx,
        "dx_frac": (dx / w) if dx is not None else None,
        "aligned": bool(dx is not None and abs(dx) < ship_width),
        "diving_near": mid_v > 400,
        "diving_dist": mid_v,
        "threat_dx_frac": (threat_dx / w) if (threat_dx is not None and mid_v > 200) else None,
        "ship_energy": ship_v,
        "pixels": (w, h),
    }