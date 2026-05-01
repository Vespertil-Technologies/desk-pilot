"""
main.py — CLI entry point for the AI mouse control tool.
"""

import argparse
import sys
import os

from dotenv import load_dotenv
from computer import BrowserSession, screenshot_grid
from agent import Agent, create_model


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

    return p.parse_args()


def main() -> None:
    # ── Load environment variables ──────────────────────────────────────────
    load_dotenv()

    args = parse_args()

    # ── Screenshot-only mode ────────────────────────────────────────────────
    if args.screenshot_only:
        print("Taking grid screenshot...")
        b64, cells = screenshot_grid(cols=args.cols, rows=args.rows)

        import base64
        import pathlib

        img_bytes = base64.b64decode(b64)
        out = pathlib.Path("grid_screenshot.png")
        out.write_bytes(img_bytes)

        print(f"Saved to {out.resolve()}  ({len(cells)} cells)")
        print(f"Sample cells: { {k: (v.x, v.y) for k, v in list(cells.items())[:5]} }")
        return

    # ── Validate goal ───────────────────────────────────────────────────────
    if not args.goal:
        print("Error: --goal is required (or use --screenshot-only)")
        sys.exit(1)

    # ── Model setup ─────────────────────────────────────────────────────────
    provider = os.getenv("MODEL_PROVIDER", "gemini")

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
    elif provider == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(f"Unknown MODEL_PROVIDER: {provider}")

    if not api_key:
        raise ValueError(f"Missing API key for provider: {provider}")

    model = create_model(provider, api_key)

    # ── Start browser session ───────────────────────────────────────────────
    browser = BrowserSession()

    try:
        if args.attach:
            print(f"Attaching to Chrome at {args.cdp_url} ...")
            browser.attach(args.cdp_url)
        else:
            print(f"Launching browser {'(headless) ' if args.headless else ''}...")
            start_url = args.url or "https://www.google.com"
            browser.start(headless=args.headless, url=start_url)

        # ── Run agent ───────────────────────────────────────────────────────
        agent = Agent(
            browser=browser,
            goal=args.goal,
            model=model,
            max_steps=args.max_steps,
        )

        agent.run(start_mode=args.mode)

    finally:
        browser.close()


if __name__ == "__main__":
    main()