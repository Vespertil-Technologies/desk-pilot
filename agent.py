"""
agent.py: the AI brain.

Each turn the agent shows the model one representation of the page
(HTML or a grid-annotated screenshot) and the model returns a single
structured action. The agent loop runs until the model returns
{"action": "done"} or hits the step limit.

Action schema (returned by the model, enforced via provider-side
structured output where supported):
    mode        "html" | "screenshot"
    action      "click" | "double_click" | "right_click"
                | "type" | "scroll" | "navigate"
                | "request_screenshot" | "request_html" | "done"
    selector    CSS selector (html mode)
    cell        grid cell name like "C4" (screenshot mode)
    text        text to type / URL to navigate
    scroll_dir  "up" | "down"
    reasoning   short explanation
Fields that don't apply to the chosen action are returned as empty strings.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from google import genai
from google.genai import types

from computer import BrowserSession, GridCell, click_cell, screenshot_grid, scroll, type_text

logger = logging.getLogger(__name__)

ACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["html", "screenshot"],
        },
        "action": {
            "type": "string",
            "enum": [
                "click",
                "double_click",
                "right_click",
                "type",
                "scroll",
                "navigate",
                "request_screenshot",
                "request_html",
                "done",
            ],
        },
        "selector": {"type": "string"},
        "cell": {"type": "string"},
        "text": {"type": "string"},
        "scroll_dir": {"type": "string", "enum": ["up", "down", ""]},
        "reasoning": {"type": "string"},
    },
    "required": [
        "mode",
        "action",
        "selector",
        "cell",
        "text",
        "scroll_dir",
        "reasoning",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a computer-use agent that controls a real browser and desktop.

Each turn you receive ONE of:
  A) The page HTML (stripped of scripts/styles), labelled [HTML MODE]
  B) A screenshot with a lettered+numbered grid overlay (A1..T15), labelled [SCREENSHOT MODE]

Reply with a single action. For fields that do not apply to your chosen
action, return an empty string.

Action semantics:
- click / double_click / right_click:
    in HTML mode set "selector"; in screenshot mode set "cell".
- type:
    in HTML mode set "selector" and "text"; in screenshot mode set "text" only.
- scroll:
    set "scroll_dir" ("up" or "down"); in screenshot mode "cell" picks the spot.
- navigate:
    set "text" to the URL.
- request_screenshot / request_html:
    switch input mode for the next turn.
- done:
    the full goal is achieved, stop.

Guidelines:
- Prefer html mode: it is cheaper and more precise.
- Switch to screenshot mode if the page uses canvas, custom widgets,
  or you can not find the element by selector.
- Always prefer the most specific CSS selector
  (id > aria-label > data-* attributes > class).
- Only return done when the full goal is achieved.

The user message will tell you whether your previous action changed the
page. If it did not, rethink and do not repeat the same action.
"""


VALID_ACTIONS = frozenset(ACTION_SCHEMA["properties"]["action"]["enum"])

# Fingerprint of the page, compared turn to turn to tell the model whether its
# last action actually did anything.
#
# Text length alone is not enough: none of typing into a field, ticking a
# checkbox, or flipping display:none changes document.body.textContent, so a
# run that was working perfectly got told "nothing happened" on every turn.
# Markup length covers style and attribute edits, and the form controls are
# folded in separately because .value and .checked never appear in the markup.
# Control values are hashed rather than kept, so passwords stay out of the
# trace and out of the next prompt.
PAGE_SIGNAL_JS = """() => {
    let formHash = 0;
    for (const el of document.querySelectorAll('input, textarea, select')) {
        const s = el.value + '\\u001f' + (el.checked ? 1 : 0);
        for (let i = 0; i < s.length; i++) {
            formHash = (formHash * 31 + s.charCodeAt(i)) | 0;
        }
    }
    const active = document.activeElement;
    return [
        document.URL,
        document.title,
        document.body ? document.body.innerHTML.length : 0,
        Math.round(window.scrollX),
        Math.round(window.scrollY),
        active ? active.tagName + '#' + (active.id || '') : '',
        formHash,
    ].join('|');
}"""


def _validate_action(action: object) -> str:
    """
    Return the action name, raising if the model handed back something the
    loop cannot execute.

    The unstructured fallback path parses whatever JSON it can find, so a
    response with no usable "action" is reachable in normal operation and must
    not take the whole run down.
    """
    if not isinstance(action, dict):
        raise ValueError(f"expected a JSON object, got {type(action).__name__}")
    act = action.get("action")
    if act not in VALID_ACTIONS:
        raise ValueError(f"missing or unknown action: {act!r}")
    return str(act)


def _extract_json(raw: str) -> dict:
    """Pull the first {...} JSON object out of a possibly-noisy response."""
    import re

    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw)
    raw = re.sub(r"```$", "", raw).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {raw[:200]}")
    return json.loads(match.group(0))


def _build_html_message(html: str, goal: str, history_summary: str, last_result: str) -> list[dict]:
    content = (
        f"GOAL: {goal}\n\n"
        f"PROGRESS SO FAR: {history_summary}\n\n"
        f"LAST ACTION RESULT: {last_result}\n\n"
        f"[HTML MODE]\n{html}"
    )
    return [{"role": "user", "content": content}]


def _build_screenshot_message(
    b64: str, goal: str, history_summary: str, last_result: str
) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"GOAL: {goal}\n\n"
                        f"PROGRESS SO FAR: {history_summary}\n\n"
                        f"LAST ACTION RESULT: {last_result}\n\n"
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


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert the internal Anthropic-shaped message list into OpenAI's format."""
    out: list[dict] = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue

        new_content = []
        for block in content:
            if block.get("type") == "text":
                new_content.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image":
                src = block["source"]
                data_url = f"data:{src['media_type']};base64,{src['data']}"
                new_content.append({"type": "image_url", "image_url": {"url": data_url}})
        out.append({"role": m["role"], "content": new_content})
    return out


class BaseModel:
    def generate(self, messages: list[dict], system: str) -> dict:
        raise NotImplementedError


class ClaudeModel(BaseModel):
    def __init__(self, api_key: str, model: str = "claude-opus-4-7"):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, messages, system):
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                tools=[
                    {
                        "name": "perform_action",
                        "description": "Perform one action and explain why.",
                        "input_schema": ACTION_SCHEMA,
                    }
                ],
                tool_choice={"type": "tool", "name": "perform_action"},
                messages=messages,
            )
            for block in resp.content:
                if block.type == "tool_use":
                    return dict(block.input)
            raise RuntimeError("Claude returned no tool_use block.")
        except Exception as structured_err:
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system,
                    messages=messages,
                )
                return _extract_json(resp.content[0].text)
            except Exception:
                raise structured_err from None


class GeminiModel(BaseModel):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _contents(self, messages, system):
        content = messages[0]["content"]
        if isinstance(content, list):
            text_part = content[0]["text"]
            image_part = content[1]["source"]["data"]
            return [
                system + "\n\n" + text_part,
                types.Part.from_bytes(
                    data=base64.b64decode(image_part),
                    mime_type="image/png",
                ),
            ]
        return system + "\n\n" + content

    def generate(self, messages, system):
        contents = self._contents(messages, system)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ACTION_SCHEMA,
                ),
            )
            return json.loads(response.text)
        except Exception as structured_err:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                )
                return _extract_json(response.text)
            except Exception:
                raise structured_err from None


class OpenAIModel(BaseModel):
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, messages, system):
        oai_messages = [{"role": "system", "content": system}] + _to_openai_messages(messages)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=oai_messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "action",
                        "schema": ACTION_SCHEMA,
                        "strict": True,
                    },
                },
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as structured_err:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=oai_messages,
                )
                return _extract_json(resp.choices[0].message.content)
            except Exception:
                raise structured_err from None


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
    def __init__(
        self,
        browser: BrowserSession,
        goal: str,
        model: BaseModel,
        max_steps: int = 20,
        trace_dir: Path | None = None,
    ) -> None:
        self.browser = browser
        self.goal = goal
        self.max_steps = max_steps
        self.model = model
        self.trace_dir = trace_dir
        self._history: list[str] = []
        self._cells: dict[str, GridCell] = {}

    def _prepare_trace(self) -> None:
        if not self.trace_dir:
            return
        (self.trace_dir / "html").mkdir(parents=True, exist_ok=True)
        (self.trace_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (self.trace_dir / "trace.jsonl").write_text("", encoding="utf-8")
        (self.trace_dir / "meta.json").write_text(
            json.dumps({"goal": self.goal, "max_steps": self.max_steps}, indent=2),
            encoding="utf-8",
        )

    def _save_payload(self, step: int, mode: str, payload: str) -> None:
        if not self.trace_dir:
            return
        if mode == "html":
            (self.trace_dir / "html" / f"step_{step:03d}.html").write_text(
                payload, encoding="utf-8"
            )
        else:
            (self.trace_dir / "screenshots" / f"step_{step:03d}.png").write_bytes(
                base64.b64decode(payload)
            )

    def _save_record(
        self,
        step: int,
        mode: str,
        action: dict | None,
        last_result: str,
        sent_to_model: str = "",
    ) -> None:
        if not self.trace_dir:
            return
        record = {
            "step": step,
            "mode": mode,
            "action": action,
            # What the model was told at the top of this turn, and what the
            # turn ended up producing. Keeping only the latter made it
            # impossible to see why the model kept retrying an action.
            "sent_to_model": sent_to_model,
            "last_result": last_result,
            "history_tail": self._history[-6:],
        }
        with (self.trace_dir / "trace.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _history_summary(self) -> str:
        if not self._history:
            return "No actions taken yet."
        return " → ".join(self._history[-6:])

    def _html_signal(self) -> str:
        return self.browser.page.evaluate(PAGE_SIGNAL_JS)

    def _check_screenshot_cell_in_window(self, cell_name: str) -> None:
        """Refuse to click a cell that falls outside the browser window."""
        bounds = self.browser.window_bounds()
        if not bounds:
            return
        cell = self._cells.get(cell_name.upper())
        if cell is None:
            return
        bx, by, bw, bh = bounds
        if not (bx <= cell.x <= bx + bw and by <= cell.y <= by + bh):
            raise ValueError(
                f"Cell {cell_name} is outside the browser window. "
                "Pick a cell that lies on the page."
            )

    def _ask(self, messages: list[dict]) -> dict:
        return self.model.generate(messages, SYSTEM_PROMPT)

    def _execute(self, action: dict) -> str:
        act = action["action"]
        mode = action.get("mode", "html")

        if act == "click":
            if mode == "html":
                sel = action["selector"]
                self.browser.click_selector(sel)
                return f"click({sel})"
            cell = action["cell"]
            self._check_screenshot_cell_in_window(cell)
            click_cell(cell, self._cells, "click")
            return f"click(cell={cell})"

        if act == "double_click":
            if mode == "html":
                sel = action["selector"]
                self.browser.double_click_selector(sel)
                return f"dblclick({sel})"
            cell = action["cell"]
            self._check_screenshot_cell_in_window(cell)
            click_cell(cell, self._cells, "double_click")
            return f"dblclick(cell={cell})"

        if act == "right_click":
            if mode == "html":
                sel = action["selector"]
                self.browser.right_click_selector(sel)
                return f"rightclick({sel})"
            cell = action["cell"]
            self._check_screenshot_cell_in_window(cell)
            click_cell(cell, self._cells, "right_click")
            return f"rightclick(cell={cell})"

        if act == "type":
            text = action.get("text", "")
            if mode == "html" and action.get("selector"):
                self.browser.type_into(action["selector"], text)
            else:
                type_text(text)
            return f"type({text!r})"

        if act == "navigate":
            url = action.get("text", "")
            self.browser.navigate(url)
            return f"navigate({url})"

        if act == "scroll":
            direction = action.get("scroll_dir") or "down"
            if mode == "html":
                self.browser.scroll_page(direction)
                return f"scroll({direction})"

            clicks = -3 if direction == "down" else 3
            cell = self._cells.get((action.get("cell") or "").upper())
            if cell is not None:
                scroll(cell.x, cell.y, clicks)
                return f"scroll({direction}, cell={cell.name})"
            if not self._cells:
                raise ValueError("cannot scroll in screenshot mode before a screenshot")
            # No cell named: scroll over the middle of the capture rather than
            # the (0, 0) screen corner, which is never over the page.
            xs = [c.x for c in self._cells.values()]
            ys = [c.y for c in self._cells.values()]
            scroll((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2, clicks)
            return f"scroll({direction})"

        if act in ("request_screenshot", "request_html", "done"):
            return f"[{act}]"

        return f"[unknown action: {act}]"

    def run(self, start_mode: str = "html") -> None:
        self._prepare_trace()

        current_mode = start_mode
        if current_mode == "screenshot" and not self.browser.can_use_screen:
            logger.warning(
                "Headless session cannot use screenshot mode. Starting in html mode."
            )
            current_mode = "html"

        last_result = "first turn, no prior action"
        prev_html_signal: str | None = None

        logger.info("Agent starting. Goal: %s", self.goal)
        if self.trace_dir:
            logger.info("Trace: %s", self.trace_dir)
        logger.info("%s", "─" * 60)

        for step in range(1, self.max_steps + 1):
            turn_mode = current_mode
            logger.info("Step %d/%d, mode: %s", step, self.max_steps, current_mode)

            if current_mode == "html":
                html = self.browser.get_html()
                signal = self._html_signal()
                if prev_html_signal is not None:
                    verdict = (
                        "page changed since the last action."
                        if signal != prev_html_signal
                        else "page did NOT change after the last action, rethink."
                    )
                    # Appended, not substituted. Overwriting threw away what the
                    # last turn actually did, including the reason a request was
                    # refused, which left the model repeating it.
                    last_result = f"{last_result} | {verdict}"
                prev_html_signal = signal
                messages = _build_html_message(
                    html, self.goal, self._history_summary(), last_result
                )
                self._save_payload(step, "html", html)
            else:
                b64, self._cells = screenshot_grid()
                prev_html_signal = None
                messages = _build_screenshot_message(
                    b64, self.goal, self._history_summary(), last_result
                )
                self._save_payload(step, "screenshot", b64)

            # Captured before the action runs and overwrites last_result. This
            # is the page-change verdict the model actually saw, which is the
            # one thing you need when working out from a trace why a run looped.
            sent_to_model = last_result

            action: dict | None = None
            try:
                action = self._ask(messages)
                act = _validate_action(action)
            except Exception as e:
                logger.error("Model error: %s", e)
                self._history.append(f"[model-error: {e}]")
                last_result = f"the last model response was unusable: {e}"
                self._save_record(step, turn_mode, action, last_result, sent_to_model)
                continue

            logger.info(
                "  → action=%r  reason=%r",
                act,
                action.get("reasoning"),
            )

            if act == "request_screenshot":
                if not self.browser.can_use_screen:
                    last_result = (
                        "screenshot mode is unavailable in this run because the"
                        " browser is headless. Stay in html mode."
                    )
                    logger.warning("  refused switch to screenshot mode (headless)")
                    self._save_record(step, turn_mode, action, last_result, sent_to_model)
                    continue
                current_mode = "screenshot"
                self._history.append("[switched→screenshot]")
                last_result = "switched to screenshot mode"
                self._save_record(step, turn_mode, action, last_result, sent_to_model)
                continue
            if act == "request_html":
                current_mode = "html"
                self._history.append("[switched→html]")
                last_result = "switched to html mode"
                self._save_record(step, turn_mode, action, last_result, sent_to_model)
                continue
            if act == "done":
                logger.info("Done after %d steps.", step)
                last_result = "done"
                self._save_record(step, turn_mode, action, last_result, sent_to_model)
                return

            try:
                desc = self._execute(action)
                self._history.append(desc)
                logger.info("  ✓ %s", desc)
                last_result = f"executed {desc}"
            except Exception as e:
                logger.warning("Action failed: %s", e)
                self._history.append(f"[error: {e}]")
                last_result = f"action failed with: {e}"
                # Falling back to screenshot mode only helps when there is a
                # window on screen to look at. Headless, it would screenshot
                # the user's desktop and click on it.
                if self.browser.can_use_screen:
                    current_mode = "screenshot"
                else:
                    last_result += " (staying in html mode, this run is headless)"

            self._save_record(step, turn_mode, action, last_result, sent_to_model)

        logger.warning("Reached max_steps (%d).", self.max_steps)
