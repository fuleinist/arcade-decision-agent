# Evidence images

These four images document the frame-capture problem in section 2 of PLAN.md. Keep them. They
are the reason the RAM path comes before the pixel path for the live page.

| file | what produced it | what it shows |
|---|---|---|
| `diag.png` | `page.screenshot()` on the full Galaga page in headless Chrome | The page loads. The game area is a black rectangle |
| `diag-canvas.png` | an element screenshot of `#canvas` in the same session | The canvas holds no game graphics |
| `frame_swiftshader.png` | an element screenshot with `--use-gl=angle --use-angle=swiftshader` | The dominant colour is `(222,222,222)`, which is the page background. Galaga uses a black starfield |
| `live_frame_headed.png` | the adapter in headed mode, after `auto_start` | The same result in a real window: the capture does not receive the WebGL surface |

Interpretation: Chrome automation does not composite this WebGL canvas on this machine. A reader
who wants to re-check the claim should capture one frame and read its dominant colour. A black
starfield means the capture works. A light grey field means it does not.