# Desk Pilot

AI agent for controlling a browser and basic desktop interactions using LLMs
(Gemini / Claude / OpenAI).

## Setup

```bash
pip install -e .
```

The first run will download Chromium automatically (~150 MB, one-time).
Run `playwright install chromium` ahead of time if you'd rather not wait.

Put your keys in a `.env` file:

```
MODEL_PROVIDER=gemini
GOOGLE_API_KEY=...
# or:
# MODEL_PROVIDER=claude
# ANTHROPIC_API_KEY=...
# MODEL_PROVIDER=openai
# OPENAI_API_KEY=...
```

Tested with `gemini-2.5-flash`, `claude-opus-4-7`, and `gpt-4o`.

A standalone environment check is available — `python test.py` exercises
PyAutoGUI, Playwright, and the configured model end-to-end with no
dependencies on the agent loop.

## Usage

Once installed, the `desk-pilot` console script is on your PATH:

```bash
desk-pilot --url "https://example.com" --goal "Click any link"
```

Equivalent without the console script:

```bash
python main.py --url "https://example.com" --goal "Click any link"
```

Common flags:

| Flag                | What it does                                                   |
| ------------------- | -------------------------------------------------------------- |
| `--url`             | URL to open at launch.                                         |
| `--goal`            | Plain-English description of what to accomplish (required).    |
| `--mode`            | `html` (default, preferred) or `screenshot`.                   |
| `--max-steps`       | Step budget before the agent gives up. Default 20.             |
| `--headless`        | Run the browser headless (no visible window).                  |
| `--attach`          | Attach to an existing Chrome started with `--remote-debugging-port=9222`. |
| `-v`, `--verbose`   | DEBUG-level logging with timestamps.                           |
| `--keep-traces`     | Keep every run's trace, not just the last one (see Traces).    |
| `--no-trace`        | Disable trace artifacts entirely.                              |
| `--screenshot-only` | Save a labeled grid screenshot to `grid_screenshot.png` and exit. |

## Modes

**HTML mode (default).** Uses Playwright. Cheap, precise, prefer for web tasks.

**Screenshot mode.** Captures the screen with a 20×15 labeled grid and uses
PyAutoGUI for clicks. Required for canvas / non-DOM widgets and for desktop
control, but less reliable.

```bash
desk-pilot --mode screenshot --goal "Open Chrome and type google.com"
```

In screenshot mode, clicks are clamped to the browser window's on-screen rect
so the agent can't accidentally hit the taskbar or window controls. The rect is
converted to physical pixels first, so the clamp stays correct on scaled
displays and under page zoom.

## Traces

By default, every run writes a trace to `~/.desk-pilot/last_run/`:

```
~/.desk-pilot/last_run/
├── meta.json           # goal, max_steps
├── trace.jsonl         # one record per turn (mode, action, last_result, history tail)
├── html/step_NNN.html  # HTML payloads sent to the model
└── screenshots/step_NNN.png   # grid screenshots (screenshot-mode turns)
```

Each run wipes the previous one, so disk usage is bounded.
`--keep-traces` writes to `~/.desk-pilot/runs/<UTC-timestamp>/` instead and
keeps history. `--no-trace` disables tracing entirely.

## Notes

- The browser launches maximized, which is required for correct click alignment.
- In screenshot mode, don't use the mouse or keyboard while the agent runs:
  PyAutoGUI moves the real cursor.
- HTML mode is preferred for web tasks. Screenshot mode is for canvas, custom
  widgets, or actual desktop apps.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The unit tests run in a couple of seconds and don't need a browser or API key.
The end-to-end tests use headless Chromium against a small data-URL fixture
page with a scripted fake model — no live network, no API spend.

CI runs both `ruff check` and `pytest` on Linux and Windows.

## Code structure

- `main.py` — CLI entry point.
- `agent.py` — model abstraction (Anthropic / Gemini / OpenAI), action schema,
  agent loop with action verification.
- `computer.py` — Playwright browser session and PyAutoGUI desktop helpers.

## Limitations

- Windows is the supported platform for actually running the agent. The other
  modules import on macOS and Linux (and the tests pass there) but desktop /
  screenshot mode hasn't been validated outside Windows.
- The agent can still misclick or loop on complex tasks. Action verification
  helps in HTML mode by telling the model when its last action didn't change
  the page, but it isn't a guarantee.
