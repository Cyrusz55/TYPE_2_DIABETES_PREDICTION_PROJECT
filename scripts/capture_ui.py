"""Capture README screenshots and an animated form demo with Playwright."""
import io
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "images"
URL = "http://127.0.0.1:8001/"
SAMPLE = {
    "no_of_pregnancy": "3",
    "age": "50",
    "bmi": "28.5",
    "glucose": "146",
    "bp_systolic": "130",
    "bp_diastolic": "85",
    "insulin": "95",
    "skin_thickness_mm": "23",
    "diabetes_pedigree_function": "0.35",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "apps.main:app", "--host", "127.0.0.1", "--port", "8001"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    frames = []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
            for _ in range(60):
                try:
                    page.goto(URL, wait_until="domcontentloaded", timeout=2000)
                    break
                except Exception:
                    time.sleep(0.5)
            else:
                raise RuntimeError("The FastAPI app did not become ready at http://127.0.0.1:8001/")

            page.screenshot(path=str(OUTPUT_DIR / "ui_empty.png"), full_page=True)
            for field, value in SAMPLE.items():
                page.locator(f"#{field}").fill(value)
                frames.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
            page.screenshot(path=str(OUTPUT_DIR / "ui_result_light.png"), full_page=True)
            frames.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))

            page.locator("#themeToggle").click()
            page.wait_for_timeout(250)
            page.screenshot(path=str(OUTPUT_DIR / "ui_result_dark.png"), full_page=True)
            page.locator("#no_of_pregnancy").fill("-1")
            page.locator("#predictBtn").click()
            page.wait_for_timeout(250)
            page.screenshot(path=str(OUTPUT_DIR / "ui_error.png"), full_page=True)
            frames.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))

            frames[0].save(
                OUTPUT_DIR / "demo.gif",
                save_all=True,
                append_images=frames[1:],
                duration=450,
                loop=0,
            )
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    main()
