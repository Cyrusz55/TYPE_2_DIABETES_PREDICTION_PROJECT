"""Capture README screenshots and an animated form demo with Playwright."""
import io
import subprocess
import sys
import time
from pathlib import Path

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


def wait_for_ready(page):
    for _ in range(60):
        try:
            page.goto(URL, wait_until="domcontentloaded", timeout=3000)
            page.wait_for_selector("#predictBtn", timeout=2000)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"The FastAPI app did not become ready at {URL}")


def capture_rgb(page, save_path: str, *, full_page: bool = False):
    return Image.open(io.BytesIO(page.screenshot(path=save_path, full_page=full_page))).convert("RGB")


def wait_for_success(page):
    page.wait_for_function(
        "document.querySelector('#results') && !document.querySelector('#results').hidden",
        timeout=20000,
    )
    page.wait_for_function(
        "document.querySelector('#formStatus') && document.querySelector('#formStatus').textContent.includes('Prediction complete.')",
        timeout=20000,
    )
    page.wait_for_timeout(900)


def create_gif(frames, save_path: Path):
    reduced = [frame.convert("P", palette=Image.ADAPTIVE, colors=32).convert("RGB") for frame in frames]
    final = reduced[-1]
    reduced = reduced + [final] * 3
    reduced[0].save(
        save_path,
        save_all=True,
        append_images=reduced[1:],
        duration=700,
        loop=0,
        optimize=True,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "apps.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1100, "height": 760}, device_scale_factor=1)
            wait_for_ready(page)

            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(OUTPUT_DIR / "ui_empty.png"), full_page=True)

            success_frames = []
            field_order = list(SAMPLE.keys())
            for i, field in enumerate(field_order):
                page.locator(f"#{field}").fill(SAMPLE[field])
                page.wait_for_timeout(120)
                success_frames.append(capture_rgb(page, str(OUTPUT_DIR / f"temp_{i}.png"), full_page=False))

            page.locator("#predictBtn").click()
            page.wait_for_function(
                "document.querySelector('#formStatus') && document.querySelector('#formStatus').textContent.includes('Sending measurements')",
                timeout=10000,
            )
            success_frames.append(capture_rgb(page, str(OUTPUT_DIR / "temp_loading.png"), full_page=False))

            wait_for_success(page)
            page.evaluate("window.scrollTo(0, 0)")
            page.locator("#results").wait_for(state="visible", timeout=20000)
            page.wait_for_timeout(900)
            page.screenshot(path=str(OUTPUT_DIR / "ui_result_light.png"), full_page=True)
            success_frames.append(capture_rgb(page, str(OUTPUT_DIR / "temp_result.png"), full_page=False))
            create_gif(success_frames, OUTPUT_DIR / "demo.gif")

            page.locator("#themeToggle").click()
            page.wait_for_timeout(350)
            page.screenshot(path=str(OUTPUT_DIR / "ui_result_dark.png"), full_page=True)

            page.locator("#resetBtn").click()
            page.locator("#no_of_pregnancy").fill("-1")
            page.locator("#predictBtn").click()
            page.wait_for_function(
                "document.querySelector('#formStatus') && document.querySelector('#formStatus').classList.contains('is-error')",
                timeout=10000,
            )
            page.screenshot(path=str(OUTPUT_DIR / "ui_error.png"), full_page=True)

            for temp_file in OUTPUT_DIR.glob("temp_*.png"):
                temp_file.unlink(missing_ok=True)

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
