"""
main.py: CLI entry point for the AI mouse control tool.
"""

import argparse
import datetime
import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent import Agent, create_model
from computer import BrowserSession, screenshot_grid

logger = logging.getLogger(__name__)

DESK_PILOT_DIR = Path.home() / ".desk-pilot"


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = (
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
        if verbose
        else "%(message)s"
    )
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


def resolve_trace_dir(no_trace: bool, keep_traces: bool) -> Path | None:
    if no_trace:
        return None
    if keep_traces:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
        d = DESK_PILOT_DIR / "runs" / ts
    else:
        d = DESK_PILOT_DIR / "last_run"
        if d.exists():
            shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI-controlled mouse/browser agent")

    p.add_argument("--url", default=None, help="URL to navigate to on launch")
    p.add_argument("--goal", default=None, help="What the agent should accomplish")

    p.add_argument(
        "--mode",
        choices=["html", "screenshot"],
        default="html",
        help="Starting capture mode (default: html)",
    )

    p.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum number of agent steps before giving up",
    )

    p.add_argument(
        "--attach",
        action="store_true",
        help="Attach to an existing Chrome (--remote-debugging-port=9222)",
    )

    p.add_argument(
        "--cdp-url",
        default="http://localhost:9222",
        help="Chrome DevTools Protocol URL (used with --attach)",
    )

    p.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )

    p.add_argument(
        "--screenshot-only",
        action="store_true",
        help="Just take a grid screenshot and save it, then exit",
    )

    p.add_argument("--cols", type=int, default=20, help="Grid columns (default 20)")
    p.add_argument("--rows", type=int, default=15, help="Grid rows (default 15)")

    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (DEBUG level with timestamps)",
    )

    trace_group = p.add_mutually_exclusive_group()
    trace_group.add_argument(
        "--keep-traces",
        action="store_true",
        help=(
            "Keep every run's trace under ~/.desk-pilot/runs/<timestamp>/."
            " Default is to overwrite ~/.desk-pilot/last_run/ each invocation."
        ),
    )
    trace_group.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable trace artifacts entirely.",
    )

    return p.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()
    configure_logging(args.verbose)

    if args.screenshot_only:
        logger.info("Taking grid screenshot...")
        b64, cells = screenshot_grid(cols=args.cols, rows=args.rows)

        import base64
        import pathlib

        img_bytes = base64.b64decode(b64)
        out = pathlib.Path("grid_screenshot.png")
        out.write_bytes(img_bytes)

        logger.info("Saved to %s  (%d cells)", out.resolve(), len(cells))
        logger.debug(
            "Sample cells: %s",
            {k: (v.x, v.y) for k, v in list(cells.items())[:5]},
        )
        return

    if not args.goal:
        logger.error("--goal is required (or use --screenshot-only)")
        sys.exit(1)

    if args.headless and args.mode == "screenshot":
        logger.error(
            "--mode screenshot needs a visible window. A headless browser paints"
            " nothing, so the grid would capture your desktop and clicks would"
            " land on it. Drop --headless or use --mode html."
        )
        sys.exit(1)

    provider = os.getenv("MODEL_PROVIDER", "gemini")

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
    elif provider == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
    else:
        raise ValueError(f"Unknown MODEL_PROVIDER: {provider}")

    if not api_key:
        raise ValueError(f"Missing API key for provider: {provider}")

    model = create_model(provider, api_key)

    browser = BrowserSession()

    try:
        if args.attach:
            logger.info("Attaching to Chrome at %s ...", args.cdp_url)
            browser.attach(args.cdp_url)
        else:
            logger.info(
                "Launching browser%s...", " (headless)" if args.headless else ""
            )
            start_url = args.url or "https://www.google.com"
            browser.start(headless=args.headless, url=start_url)

        # ── Run agent ───────────────────────────────────────────────────────
        trace_dir = resolve_trace_dir(args.no_trace, args.keep_traces)

        agent = Agent(
            browser=browser,
            goal=args.goal,
            model=model,
            max_steps=args.max_steps,
            trace_dir=trace_dir,
        )

        agent.run(start_mode=args.mode)

    finally:
        browser.close()


if __name__ == "__main__":
    main()