"""Page adapter: drive a browser-hosted arcade game through Playwright.

Written against classicgamesarcade.com's Galaga (2026-10-16), which runs archive.org's
**Emularity** loader with a MAME WebAssembly core. That matters, because Emularity does
not hide anything - verified live on that page:

  window.Module                      the Emscripten module; `_SDL_SendKeyboardKey` is
                                     callable, `HEAPU8` is 29 MB of readable RAM, and
                                     `cwrap` is present
  Module.__ZN15running_machine15emscripten_saveEPKc / ...loadEPKc / ...soft_resetEv
                                     real MAME save states, load states and reset -
                                     reachable from page JS, which is what makes
                                     reproducible episodes possible at all
  window.onkeydown                   Emularity binds it (loader.js:1900), so real
                                     browser key events reach the emulator
  canvas#canvas, 672x768, WebGL      getContext('2d') returns null here, so the frame is
                                     taken with an element screenshot, not toDataURL

Pitfalls measured on the live page, all handled below:
  * the canvas sits ~600 px down a 2800 px page; an unscrolled clip screenshot is a
    2 KB mostly-black PNG, so the element must be scrolled into view first;
  * the page carries AdSense/analytics/recaptcha iframes and a cookie notice - the
    adapter never reads or clicks page chrome, only the canvas element;
  * frames are ~8-9 KB PNGs at 672x768, so a 30 Hz loop is ~250 KB/s if you stream them.
"""
from __future__ import annotations

import base64
import os
import time
from typing import Optional

from ..obs import Action, GameEnv, Obs
from ..perceive import frame_features, prose_from_features, threat_prose

DEFAULT_URL = "https://www.classicgamesarcade.com/game/21637/galaga.html"

# The page's own control list: "Key 5 to add credits. 1 to start. Arrows to move.
# Ctrl to fire." (Space is *not* fire in this MAME build - measured 2026-10-16.)
KEYS = {
    "left": ["ArrowLeft"], "right": ["ArrowRight"],
    "fire": ["Control"], "left_fire": ["ArrowLeft", "Control"], "right_fire": ["ArrowRight", "Control"],
    "hold": [],
    "coin": ["5"], "start": ["1"], "reset": ["F3"],
}
ACTION_SPACE = ["hold", "left", "right", "fire", "left_fire", "right_fire"]


class EmularityEnv(GameEnv):
    name = "page"

    def __init__(self, url: Optional[str] = None, headed: bool = False, hold_ms: int = 60,
                 viewport: tuple[int, int] = (1440, 1100), boot_timeout_s: int = 90,
                 auto_start: bool = True, cooldown_s: float = 0.4,
                 channel: str = "chrome", headless_shell: bool = False) -> None:
        self.url = url or os.environ.get("ARCADE_URL", DEFAULT_URL)
        self.headed = headed
        self.hold_ms = hold_ms
        self.viewport = viewport
        self.boot_timeout_s = boot_timeout_s
        self.auto_start = auto_start
        self.cooldown_s = cooldown_s
        #: "chrome" uses the installed Chrome (falls back to bundled chromium);
        #: headless_shell=True forces the bundled headless shell instead.
        self.channel = channel
        self.headless_shell = headless_shell
        self.action_space = ACTION_SPACE
        self._pw = None
        self._browser = None
        self._page = None
        self._frame_i = 0

    # ---------------------------------------------------------------- lifecycle ---
    def _ensure(self) -> None:
        if self._page:
            return
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        args = ["--autoplay-policy=no-user-gesture-required", "--mute-audio",
                "--disable-blink-features=AutomationControlled"]
        # Prefer a real Chrome install when there is one: ad-heavy pages behave better and
        # the browser is the one the user actually tests in. Fall back to bundled chromium.
        self._browser = None
        if not self.headless_shell:
            try:
                self._browser = self._pw.chromium.launch(headless=not self.headed,
                                                         channel=self.channel, args=args)
            except Exception:
                self._browser = None
        if self._browser is None:
            self._browser = self._pw.chromium.launch(headless=not self.headed, args=args)
        ctx = self._browser.new_context(viewport={"width": self.viewport[0], "height": self.viewport[1]},
                                        device_scale_factor=1)
        self._page = ctx.new_page()
        self._page.goto(self.url, wait_until="domcontentloaded", timeout=self.boot_timeout_s * 1000)
        self._page.wait_for_selector("#canvas", timeout=self.boot_timeout_s * 1000)
        deadline = time.time() + self.boot_timeout_s
        while time.time() < deadline:
            ready = self._page.evaluate("() => !!(window.Module && window.Module.calledRun)")
            if ready:
                break
            time.sleep(0.5)
        self._page.evaluate("""() => { const c=document.getElementById('canvas'); if (c) c.scrollIntoView({block:'center'}); }""")
        if not self._wait_for_video():
            raise RuntimeError("the emulator loaded but never produced a non-blank, moving frame")
        if self.auto_start:
            self.insert_coin_and_start()

    def _wait_for_video(self, timeout_s: int = 45) -> bool:
        """`calledRun` is not enough: MAME reports ready several seconds before the ROM
        produces video, and an element screenshot of a live WebGL canvas is black until
        it does. Wait for a frame that is neither blank nor still."""
        from ..perceive import blank_fraction

        prev, prev_age, stable = None, 0.0, 0
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            png = self.frame()
            if png:
                blank = blank_fraction(png)
                moved = prev is not None and png != prev
                if blank < 0.9 and (moved or stable >= 2):
                    return True
                stable = stable + 1 if not moved else 0
                prev = png
            time.sleep(0.5)
        return False

    def insert_coin_and_start(self) -> None:
        for k in ("coin", "start"):
            self._tap(k)
            time.sleep(0.8)

    def _tap(self, name: str) -> None:
        for key in KEYS[name]:
            self._page.keyboard.press(key)

    def reset(self) -> Obs:
        self._ensure()
        # a save state would be the reproducible way here; today we reload, which is
        # cheap on this page and keeps the adapter honest about what it can guarantee
        self._page.reload(wait_until="domcontentloaded")
        self._page.wait_for_selector("#canvas", timeout=self.boot_timeout_s * 1000)
        deadline = time.time() + self.boot_timeout_s
        while time.time() < deadline:
            if self._page.evaluate("() => !!(window.Module && window.Module.calledRun)"):
                break
            time.sleep(0.5)
        self._page.evaluate("""() => { const c=document.getElementById('canvas'); if (c) c.scrollIntoView({block:'center'}); }""")
        if not self._wait_for_video():
            raise RuntimeError("the emulator reloaded but produced no video")
        if self.auto_start:
            self.insert_coin_and_start()
        return self._obs(done=False)

    # --------------------------------------------------------------------- step ---
    def step(self, action: Action) -> Obs:
        self._ensure()
        keys = KEYS.get(action.name, [])
        for k in keys:
            self._page.keyboard.down(k)
        if keys and self.hold_ms:
            time.sleep(self.hold_ms / 1000.0)
        for k in keys:
            self._page.keyboard.up(k)
        if self.cooldown_s:
            time.sleep(self.cooldown_s)
        self._frame_i += 1
        return self._obs(done=False)

    def _obs(self, done: bool) -> Obs:
        png = self.frame()
        feats = frame_features(png) if png else {}
        feats["threat_dx_frac"] = None
        return Obs(frame=png, features=feats,
                   prose=prose_from_features(feats) + " " + threat_prose(feats),
                   score=0, done=done)

    def frame(self) -> Optional[bytes]:
        self._ensure()
        try:
            el = self._page.query_selector("#canvas")
            if not el:
                return None
            return el.screenshot(type="png")
        except Exception:
            return None

    # ---------------------------------------------------------------- MAME hooks ---
    def read_ram(self, addr: int, length: int = 1) -> Optional[bytes]:
        """Read the wasm heap. Address mapping to MAME's Z80 space is not done yet."""
        self._ensure()
        try:
            vals = self._page.evaluate(
                "([a, n]) => { const m = window.Module; if (!m || !m.HEAPU8) return null;"
                " return Array.from(m.HEAPU8.subarray(a, a + n)); }", [addr, length])
            return bytes(vals) if vals else None
        except Exception:
            return None

    def _write_cstring(self, text: str) -> int:
        """Emscripten renamed allocateUTF8 away; write the bytes ourselves."""
        return self._page.evaluate(
            """(s) => { const m = window.Module; const bytes = new TextEncoder().encode(s);
                       const p = m._malloc(bytes.length + 1);
                       m.HEAPU8.set(bytes, p); m.HEAPU8[p + bytes.length] = 0; return p; }""", text)

    def _call_mame(self, symbol: str, name: str) -> bool:
        self._ensure()
        try:
            return bool(self._page.evaluate(
                """([sym, n]) => { const m = window.Module; const f = m[sym];
                     if (typeof f !== 'function') return false;
                     const bytes = new TextEncoder().encode(n);
                     const p = m._malloc(bytes.length + 1);
                     m.HEAPU8.set(bytes, p); m.HEAPU8[p + bytes.length] = 0;
                     const r = f(p); m._free(p); return r !== false; }""",
                [symbol, name]))
        except Exception:
            return False

    def mame_save(self, name: str = "q0") -> bool:
        """MAME save state via running_machine::emscripten_save(char const*)."""
        return self._call_mame("__ZN15running_machine15emscripten_saveEPKc", name)

    def mame_load(self, name: str = "q0") -> bool:
        return self._call_mame("__ZN15running_machine15emscripten_loadEPKc", name)

    def mame_soft_reset(self) -> bool:
        return self._call_mame("__ZN15running_machine21emscripten_soft_resetEv", "")

    def close(self) -> None:
        for closer in (self._browser, self._pw):
            try:
                if closer:
                    closer.close() if closer is self._browser else closer.stop()
            except Exception:
                pass
        self._page = None