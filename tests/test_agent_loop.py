"""
End-to-end coverage for the loop's reliability machinery: page-change
detection, scroll dispatch, malformed model output, and the headless guard.

Uses headless Chromium and a scripted model, so no API key and no spend.
"""

import json
import re
import urllib.parse

import pytest

from agent import Agent, BaseModel, _validate_action
from computer import BrowserSession

FORM_FIXTURE = """
<!DOCTYPE html>
<html><body>
  <input id='name' type='text' />
  <input id='agree' type='checkbox' />
  <button id='show' onclick="document.getElementById('panel').style.display='block'">Show</button>
  <div id='panel' style='display:none'>panel</div>
  <div style='height: 4000px'>tall</div>
</body></html>
"""


def _action(**kwargs):
    base = {
        "mode": "html",
        "action": "done",
        "selector": "",
        "cell": "",
        "text": "",
        "scroll_dir": "",
        "reasoning": "",
    }
    base.update(kwargs)
    return base


class RecordingModel(BaseModel):
    """Returns scripted actions and keeps the LAST ACTION RESULT it was shown."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = 0
        self.seen_results: list[str] = []

    def generate(self, messages, system):
        content = messages[0]["content"]
        text = content if isinstance(content, str) else content[0]["text"]
        self.seen_results.append(re.search(r"LAST ACTION RESULT: (.*)", text).group(1))
        if self.calls >= len(self.actions):
            return _action(action="done", reasoning="out of script")
        action = self.actions[self.calls]
        self.calls += 1
        return action


@pytest.fixture
def browser():
    url = "data:text/html;charset=utf-8," + urllib.parse.quote(FORM_FIXTURE)
    session = BrowserSession()
    session.start(headless=True, url=url)
    yield session
    session.close()


def _run(browser, actions, **kwargs):
    model = RecordingModel(actions)
    Agent(browser=browser, goal="probe", model=model, max_steps=8, **kwargs).run()
    return model


# ── page-change detection ────────────────────────────────────────────────────


def test_typing_is_reported_as_a_change(browser):
    """Typing leaves textContent untouched, so the old signal missed it."""
    model = _run(browser, [_action(action="type", selector="#name", text="Arav")])
    assert browser.page.eval_on_selector("#name", "el => el.value") == "Arav"
    assert "did NOT change" not in model.seen_results[1]


def test_ticking_a_checkbox_is_reported_as_a_change(browser):
    model = _run(browser, [_action(action="click", selector="#agree")])
    assert browser.page.eval_on_selector("#agree", "el => el.checked") is True
    assert "did NOT change" not in model.seen_results[1]


def test_revealing_a_hidden_panel_is_reported_as_a_change(browser):
    model = _run(browser, [_action(action="click", selector="#show")])
    assert browser.page.eval_on_selector("#panel", "el => el.style.display") == "block"
    assert "did NOT change" not in model.seen_results[1]


def test_a_genuine_no_op_still_reports_no_change(browser):
    """The signal must not become so noisy that it never fires."""
    model = _run(browser, [_action(action="scroll", scroll_dir="up")])  # already at top
    assert "did NOT change" in model.seen_results[1]


# ── scroll dispatch ──────────────────────────────────────────────────────────


def test_html_mode_scroll_moves_the_page_not_the_mouse(browser, monkeypatch):
    """HTML mode must never reach for PyAutoGUI, which drives the real cursor."""
    import agent as agent_module

    def fail(*args, **kwargs):
        raise AssertionError("html-mode scroll used the desktop mouse")

    monkeypatch.setattr(agent_module, "scroll", fail)

    _run(browser, [_action(action="scroll", scroll_dir="down")])
    assert browser.page.evaluate("() => window.scrollY") > 0


# ── malformed model output ───────────────────────────────────────────────────


def test_response_without_an_action_key_does_not_crash(browser):
    model = RecordingModel([])
    model.generate = lambda messages, system: {"msg": "hello"}
    agent = Agent(browser=browser, goal="probe", model=model, max_steps=2)
    agent.run()  # must exhaust its steps, not raise
    assert any("model-error" in h for h in agent._history)


def test_unknown_action_name_is_rejected():
    with pytest.raises(ValueError, match="unknown action"):
        _validate_action(_action(action="teleport"))


def test_non_object_response_is_rejected():
    with pytest.raises(ValueError, match="expected a JSON object"):
        _validate_action(["not", "a", "dict"])


def test_valid_action_returns_its_name():
    assert _validate_action(_action(action="click")) == "click"


# ── headless guard ───────────────────────────────────────────────────────────


def test_failed_action_stays_in_html_mode_when_headless(browser, monkeypatch):
    """Screenshot fallback headless would grab the user's desktop."""
    import agent as agent_module

    def fail(*args, **kwargs):
        raise AssertionError("headless run tried to screenshot the desktop")

    monkeypatch.setattr(agent_module, "screenshot_grid", fail)

    model = _run(browser, [_action(action="click", selector="#missing")])
    assert "staying in html mode" in model.seen_results[-1] or model.calls >= 1


def test_request_screenshot_is_refused_when_headless(browser, monkeypatch):
    import agent as agent_module

    monkeypatch.setattr(
        agent_module,
        "screenshot_grid",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("screenshotted desktop")),
    )

    model = _run(browser, [_action(action="request_screenshot")])
    assert "unavailable" in model.seen_results[1]


def test_headless_session_reports_no_screen(browser):
    assert browser.can_use_screen is False


# ── trace ────────────────────────────────────────────────────────────────────


def test_trace_records_what_the_model_was_told(browser, tmp_path):
    _run(
        browser,
        [_action(action="type", selector="#name", text="Arav"), _action(action="done")],
        trace_dir=tmp_path,
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [r["sent_to_model"] for r in records][0] == "first turn, no prior action"
    assert all("sent_to_model" in r for r in records)
