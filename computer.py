"""
computer.py — low-level interface to the screen, browser, and mouse.

Two capture modes:
  1. get_html()        → returns cleaned HTML string (cheap, text-based)
  2. screenshot_grid() → returns base64 PNG with lettered/numbered grid overlay

Two action modes:
  1. click_selector(css)      → clicks an element by CSS selector (Playwright)
  2. click_grid(cell, action) → maps "C4" → pixel coords and clicks (PyAutoGUI)
"""

from __future__ import annotations

import base64
import io
import string
import sys
import time
from dataclasses import dataclass
from typing import Literal

import pyautogui
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Browser, Page, sync_playwright

if sys.platform == "win32":
    import ctypes
    ctypes.windll.user32.SetProcessDPIAware()


# ── Grid config ──────────────────────────────────────────────────────────────

GRID_COLS = 20          # A-T (letters)
GRID_ROWS = 15          # 1-15 (numbers)
GRID_LINE_COLOR  = (255, 80, 80, 120)   # semi-transparent red
GRID_LABEL_COLOR = (255, 60, 60, 220)   # slightly more opaque
GRID_FONT_SIZE   = 14

COLUMNS = list(string.ascii_uppercase[:GRID_COLS])   # ['A', 'B', ..., 'T']


@dataclass
class GridCell:
    col: str    # e.g. "C"
    row: int    # e.g. 4
    x: int      # pixel center x
    y: int      # pixel center y

    @property
    def name(self) -> str:
        return f"{self.col}{self.row}"


# ── Screenshot + grid ─────────────────────────────────────────────────────────

def screenshot_grid(
    region: tuple[int, int, int, int] | None = None,
    cols: int = GRID_COLS,
    rows: int = GRID_ROWS,
) -> tuple[str, dict[str, GridCell]]:
    """
    Take a screenshot, overlay a labeled grid, return:
      - base64-encoded PNG string
      - dict mapping cell name → GridCell (so the agent can resolve coords later)

    region: (left, top, width, height) or None for full screen
    """
    img = pyautogui.screenshot(region=region)
    img = img.convert("RGBA")
    W, H = img.size

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    cell_w = W / cols
    cell_h = H / rows

    font = None
    for path in (
        "consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ):
        try:
            font = ImageFont.truetype(path, GRID_FONT_SIZE)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    cells: dict[str, GridCell] = {}

    for ci, col_letter in enumerate(COLUMNS[:cols]):
        for ri in range(rows):
            row_num = ri + 1
            cx = int(cell_w * ci + cell_w / 2)
            cy = int(cell_h * ri + cell_h / 2)

            cell = GridCell(col=col_letter, row=row_num, x=cx, y=cy)
            cells[cell.name] = cell

            # Vertical line at left edge of cell
            if ci > 0:
                x_line = int(cell_w * ci)
                draw.line([(x_line, 0), (x_line, H)], fill=GRID_LINE_COLOR, width=1)

            # Horizontal line at top edge of cell
            if ri > 0:
                y_line = int(cell_h * ri)
                draw.line([(0, y_line), (W, y_line)], fill=GRID_LINE_COLOR, width=1)

            # Cell label in top-left corner of cell
            label_x = int(cell_w * ci) + 3
            label_y = int(cell_h * ri) + 2
            draw.text((label_x, label_y), cell.name, font=font, fill=GRID_LABEL_COLOR)

    combined = Image.alpha_composite(img, overlay).convert("RGB")

    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return b64, cells


# ── HTML capture ─────────────────────────────────────────────────────────────

class BrowserSession:
    """Thin wrapper around a Playwright browser session."""

    def __init__(self) -> None:
        self._pw = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    def start(self, headless: bool = False, url: str | None = None) -> None:
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=headless,
            args=["--start-maximized"]
        )
        self._page = self._browser.new_page(no_viewport=True)
        if url:
            self._page.goto(url, wait_until="domcontentloaded")

    def attach(self, cdp_url: str = "http://localhost:9222") -> None:
        """Attach to an already-running Chrome with --remote-debugging-port=9222."""
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.connect_over_cdp(cdp_url)
        ctx = self._browser.contexts[0]
        self._page = ctx.pages[0] if ctx.pages else ctx.new_page()

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("BrowserSession not started. Call .start() or .attach() first.")
        return self._page

    def get_html(self, max_chars: int = 60_000) -> str:
        """
        Return cleaned HTML of the current page.
        Strips <script>, <style>, <svg> blobs to keep token count low.
        """
        html: str = self.page.evaluate("""() => {
            const clone = document.documentElement.cloneNode(true);
            // Remove script/style/svg to slim the payload
            for (const tag of ['script', 'style', 'svg', 'noscript']) {
                clone.querySelectorAll(tag).forEach(el => el.remove());
            }
            // Keep aria-label, data-testid for the AI to target elements
            return clone.outerHTML;
        }""")
        if len(html) > max_chars:
            html = html[:max_chars] + "\n<!-- [truncated] -->"
        return html

    def navigate(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded")

    def click_selector(self, selector: str) -> None:
        """Click an element by CSS selector."""
        self.page.click(selector)
        time.sleep(0.3)   # brief settle

    def double_click_selector(self, selector: str) -> None:
        self.page.dblclick(selector)
        time.sleep(0.3)

    def type_into(self, selector: str, text: str) -> None:
        self.page.fill(selector, text)

    def screenshot_b64(self) -> str:
        """Take a Playwright screenshot (no grid) and return as base64."""
        buf = self.page.screenshot(type="png")
        return base64.b64encode(buf).decode()

    def close(self) -> None:
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()


# ── Mouse / desktop actions ───────────────────────────────────────────────────

def move_to(x: int, y: int, duration: float = 0.3) -> None:
    pyautogui.moveTo(x, y, duration=duration)

def click_at(x: int, y: int) -> None:
    pyautogui.click(x, y)
    time.sleep(0.2)

def double_click_at(x: int, y: int) -> None:
    pyautogui.doubleClick(x, y)
    time.sleep(0.2)

def right_click_at(x: int, y: int) -> None:
    pyautogui.rightClick(x, y)
    time.sleep(0.2)

def type_text(text: str) -> None:
    pyautogui.typewrite(text, interval=0.05)

def press_key(key: str) -> None:
    pyautogui.press(key)

def scroll(x: int, y: int, clicks: int) -> None:
    """Positive clicks = scroll up, negative = scroll down."""
    pyautogui.scroll(clicks, x=x, y=y)


def click_cell(
    cell_name: str,
    cells: dict[str, GridCell],
    action: Literal["click", "double_click", "right_click"] = "click",
) -> None:
    """
    Given a grid cell name like "C4", look up pixel coords and click.
    cells must be the dict returned by screenshot_grid().
    """
    cell_name = cell_name.strip().upper()
    if cell_name not in cells:
        raise ValueError(f"Cell '{cell_name}' not found. Valid cells: {list(cells.keys())[:10]}...")
    cell = cells[cell_name]
    if action == "click":
        click_at(cell.x, cell.y)
    elif action == "double_click":
        double_click_at(cell.x, cell.y)
    elif action == "right_click":
        right_click_at(cell.x, cell.y)