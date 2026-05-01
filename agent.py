"""
agent.py — the AI brain.

Sends HTML or a grid-annotated screenshot to Claude and parses back a structured
action.  The agent loop runs until the AI returns {"action": "done"} or hits the
step limit.

Action schema (JSON, returned by the model):
{
  "mode":      "html" | "screenshot",     // which mode was used this turn
  "action":    "click" | "double_click" | "right_click"
               | "type" | "scroll" | "navigate"
               | "request_screenshot"     // switch to screenshot mode next turn
               | "request_html"           // switch to html mode next turn
               | "done",
  "selector":  "<css selector>",          // used when mode=html
  "cell":      "C4",                      // used when mode=screenshot
  "text":      "...",                     // for type / navigate actions
  "scroll_dir": "up" | "down",           // for scroll action
  "reasoning": "why I chose this"        // always present
}
"""

from __future__ import annotations
import json
import re
from typing import Any
import base64
import anthropic
from google import genai
from google.genai import types

from computer import BrowserSession, GridCell, screenshot_grid, click_cell, click_at, \
    double_click_at, right_click_at, type_text, press_key, scroll

MODEL = "claude-opus-4-5"

SYSTEM_PROMPT = """You are a computer-use agent that controls a real browser and desktop.

On each turn you receive EITHER:
  A) The page HTML (stripped of scripts/styles), labelled [HTML MODE]
  B) A screenshot with a lettered+numbered grid overlay (A1..T15), labelled [SCREENSHOT MODE]

You must reply with a SINGLE JSON object — no markdown fences, nothing else — using this schema:

{
  "mode":      "html" | "screenshot",
  "action":    one of:
                 "click"              → click an element or grid cell
                 "double_click"       → double-click
                 "right_click"        → right-click
                 "type"               → type text (selector or cell required first)
                 "scroll"             → scroll at a cell location
                 "navigate"           → go to a URL (text = URL)
                 "request_screenshot" → you need to see the screen visually
                 "request_html"       → you want the lighter HTML representation
                 "done"               → goal achieved, stop
  "selector":  "CSS selector"         (only in html mode)
  "cell":      "B3"                   (only in screenshot mode)
  "text":      "string"               (for type / navigate)
  "scroll_dir": "up" | "down"        (for scroll)
  "reasoning": "brief explanation"
}

Guidelines:
- Prefer html mode: it is cheaper and more precise.
- Switch to screenshot mode if the page uses canvas, custom widgets, or you can not find the element by selector.
- Always prefer the most specific CSS selector (id > aria-label > data-* attributes > class).
- If the action succeeded but there is more to do, continue with the next action.
- Set action=done only when the full goal is achieved.
"""


def _extract_json(raw: str) -> dict:
    import re, json

    raw = raw.strip()

    # Remove markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw)
    raw = re.sub(r"```$", "", raw).strip()

    # Extract JSON block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {raw[:200]}")

    return json.loads(match.group(0))


def _build_html_message(html: str, goal: str, history_summary: str) -> list[dict]:
    content = (
        f"GOAL: {goal}\n\n"
        f"PROGRESS SO FAR: {history_summary}\n\n"
        f"[HTML MODE]\n{html}"
    )
    return [{"role": "user", "content": content}]


def _build_screenshot_message(b64: str, goal: str, history_summary: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"GOAL: {goal}\n\n"
                        f"PROGRESS SO FAR: {history_summary}\n\n"
                        "[SCREENSHOT MODE] The image below has a red grid overlay. "
                        "Columns are labelled A-T (left→right), rows 1-15 (top→bottom). "
                        "Identify the best cell and action to take."
                    ),
                },
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": b64},
                },
            ],
        }
    ]

class BaseModel:
    def generate(self, messages: list[dict], system: str) -> str:
        raise NotImplementedError


class ClaudeModel(BaseModel):
    def __init__(self, api_key: str, model: str = "claude-opus-4-5"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, messages, system):
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=system,
            messages=messages,
        )
        return resp.content[0].text


class GeminiModel(BaseModel):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, messages, system):
        content = messages[0]["content"]

        # ── Screenshot mode (multimodal) ───────────────────────────
        if isinstance(content, list):
            text_part = content[0]["text"]
            image_part = content[1]["source"]["data"]

            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    system + "\n\n" + text_part,
                    types.Part.from_bytes(
                        data=base64.b64decode(image_part),
                        mime_type="image/png"
                    )
                ]
            )

        # ── HTML mode (text only) ──────────────────────────────────
        else:
            response = self.client.models.generate_content(
                model=self.model,
                contents=system + "\n\n" + content
            )

        return response.text


class OpenAIModel(BaseModel):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, messages, system):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages,
        )
        return resp.choices[0].message.content


def create_model(provider: str, api_key: str) -> BaseModel:
    if provider == "claude":
        return ClaudeModel(api_key)
    elif provider == "gemini":
        return GeminiModel(api_key)
    elif provider == "openai":
        return OpenAIModel(api_key)
    else:
        raise ValueError(provider)
class Agent:
    def __init__(self, browser: BrowserSession, goal: str,model: BaseModel, max_steps: int = 20) -> None:
        self.browser = browser
        self.goal = goal
        self.max_steps = max_steps
        self.model = model   # inject dependency
        self._history: list[str] = []   # short action log for context
        self._cells: dict[str, GridCell] = {}   # updated each screenshot turn

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _history_summary(self) -> str:
        if not self._history:
            return "No actions taken yet."
        return " → ".join(self._history[-6:])   # last 6 steps

    def _ask(self, messages: list[dict]) -> dict:
        raw = self.model.generate(messages, SYSTEM_PROMPT)
        return _extract_json(raw)

    def _execute(self, action: dict) -> str:
        """Execute one action dict. Returns a short description for the history log."""
        act = action["action"]
        mode = action.get("mode", "html")
        reason = action.get("reasoning", "")

        if act == "click":
            if mode == "html":
                sel = action["selector"]
                self.browser.click_selector(sel)
                return f"click({sel})"
            else:
                cell = action["cell"]
                click_cell(cell, self._cells, "click")
                return f"click(cell={cell})"

        elif act == "double_click":
            if mode == "html":
                sel = action["selector"]
                self.browser.double_click_selector(sel)
                return f"dblclick({sel})"
            else:
                cell = action["cell"]
                click_cell(cell, self._cells, "double_click")
                return f"dblclick(cell={cell})"

        elif act == "right_click":
            if mode == "html":
                # Playwright doesn't have a simple rightClick helper, use JS
                sel = action["selector"]
                self.browser.page.click(sel, button="right")
                return f"rightclick({sel})"
            else:
                cell = action["cell"]
                click_cell(cell, self._cells, "right_click")
                return f"rightclick(cell={cell})"

        elif act == "type":
            text = action.get("text", "")
            if mode == "html" and action.get("selector"):
                self.browser.type_into(action["selector"], text)
            else:
                type_text(text)
            return f"type({text!r})"

        elif act == "navigate":
            url = action.get("text", "")
            self.browser.navigate(url)
            return f"navigate({url})"

        elif act == "scroll":
            direction = action.get("scroll_dir", "down")
            clicks = -3 if direction == "down" else 3
            if mode == "screenshot" and action.get("cell"):
                c = self._cells.get(action["cell"].upper())
                if c:
                    scroll(c.x, c.y, clicks)
            else:
                scroll(0, 0, clicks)   # scroll wherever mouse is
            return f"scroll({direction})"

        elif act in ("request_screenshot", "request_html", "done"):
            return f"[{act}]"

        else:
            return f"[unknown action: {act}]"

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, start_mode: str = "html") -> None:
        """
        Run the agent loop until done or max_steps reached.
        start_mode: "html" | "screenshot"
        """
        current_mode = start_mode
        print(f"\n🤖 Agent starting. Goal: {self.goal}\n{'─'*60}")

        for step in range(1, self.max_steps + 1):
            print(f"\nStep {step}/{self.max_steps} — mode: {current_mode}")

            # ── Build the prompt ──────────────────────────────────────────────
            if current_mode == "html":
                html = self.browser.get_html()
                messages = _build_html_message(html, self.goal, self._history_summary())
            else:
                b64, self._cells = screenshot_grid()
                messages = _build_screenshot_message(b64, self.goal, self._history_summary())

            # ── Ask Claude ────────────────────────────────────────────────────
            action = self._ask(messages)
            print(f"  ↳ action={action.get('action')!r}  reason={action.get('reasoning')!r}")

            # ── Handle mode switches ──────────────────────────────────────────
            if action["action"] == "request_screenshot":
                current_mode = "screenshot"
                self._history.append("[switched→screenshot]")
                continue

            if action["action"] == "request_html":
                current_mode = "html"
                self._history.append("[switched→html]")
                continue

            if action["action"] == "done":
                print(f"\n✅ Done after {step} steps.")
                return

            # ── Execute ───────────────────────────────────────────────────────
            try:
                desc = self._execute(action)
                self._history.append(desc)
                print(f"  ✓ {desc}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self._history.append(f"[error: {e}]")
                # On error, switch to screenshot for a fresh look
                current_mode = "screenshot"

        print(f"\n⚠️  Reached max_steps ({self.max_steps}).")