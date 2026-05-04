"""
End-to-end integration tests against a real (headless) browser.

These tests use Playwright's headless Chromium and a scripted "model"
that returns pre-baked actions, so they exercise the whole agent loop
without needing API keys.
"""

import urllib.parse

import pytest

from agent import Agent, BaseModel
from computer import BrowserSession


def _action(**kwargs):
    """Build a complete action dict (every schema field present)."""
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


class ScriptedModel(BaseModel):
    """A BaseModel that returns a fixed sequence of actions, then 'done'."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = 0

    def generate(self, messages, system):
        if self.calls >= len(self.actions):
            return _action(action="done", reasoning="out of script")
        action = self.actions[self.calls]
        self.calls += 1
        return action


HTML_FIXTURE = """
<!DOCTYPE html>
<html><body>
  <button id='go' onclick="document.title='clicked'">Go</button>
  <input id='name' type='text' />
</body></html>
"""


@pytest.fixture
def browser():
    fixture_url = "data:text/html;charset=utf-8," + urllib.parse.quote(HTML_FIXTURE)
    session = BrowserSession()
    session.start(headless=True, url=fixture_url)
    yield session
    session.close()


def test_click_in_html_mode(browser, tmp_path):
    model = ScriptedModel(
        [
            _action(action="click", selector="#go", reasoning="click button"),
            _action(action="done", reasoning="done"),
        ]
    )
    agent = Agent(
        browser=browser,
        goal="click the button",
        model=model,
        max_steps=5,
        trace_dir=tmp_path,
    )
    agent.run()

    assert browser.page.title() == "clicked"
    assert model.calls == 2
    assert (tmp_path / "trace.jsonl").exists()
    assert (tmp_path / "meta.json").exists()


def test_type_in_html_mode(browser):
    model = ScriptedModel(
        [
            _action(
                action="type",
                selector="#name",
                text="Arav",
                reasoning="enter name",
            ),
            _action(action="done", reasoning="done"),
        ]
    )
    agent = Agent(
        browser=browser,
        goal="type a name",
        model=model,
        max_steps=5,
    )
    agent.run()

    value = browser.page.eval_on_selector("#name", "el => el.value")
    assert value == "Arav"


def test_done_terminates_loop_immediately(browser):
    model = ScriptedModel([_action(action="done", reasoning="nothing to do")])
    agent = Agent(
        browser=browser,
        goal="bail",
        model=model,
        max_steps=10,
    )
    agent.run()

    assert model.calls == 1
