"""Screenshot every [data-shot] element of build/figs.html and build/eqs.html.

Writes build/img/<name>.png at 4x and build/img/manifest.json with each
element's CSS size, so build.js can place it at size / 96 inches.
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BUILD = Path(sys.argv[1]).resolve()
IMG = BUILD / "img"
IMG.mkdir(parents=True, exist_ok=True)
manifest, errors = {}, []

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(device_scale_factor=4, viewport={"width": 1600, "height": 1200})
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    for html, flag, transparent in [("figs.html", "FIGS_DONE", False), ("eqs.html", "EQS_DONE", True)]:
        page.goto((BUILD / html).as_uri())
        page.wait_for_function(f"window.{flag} === true", timeout=60000)
        for el in page.query_selector_all("[data-shot]"):
            name = el.get_attribute("data-shot")
            box = el.bounding_box()
            el.screenshot(path=str(IMG / f"{name}.png"), omit_background=transparent)
            manifest[name] = {"w": round(box["width"], 2), "h": round(box["height"], 2)}
    browser.close()

(IMG / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(len(manifest), "images;", "errors:", errors or "none")
