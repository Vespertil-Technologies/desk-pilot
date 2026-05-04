import ctypes
import os
import time

from dotenv import load_dotenv

load_dotenv()
print("\n=== BASIC ENV CHECK ===")

# ── 1. DPI FIX (Windows) ─────────────────────────────────────────
try:
    ctypes.windll.user32.SetProcessDPIAware()
    print("[✓] DPI awareness set")
except Exception as e:
    print("[✗] DPI fix failed:", e)


# ── 2. PyAutoGUI TEST ────────────────────────────────────────────
print("\n=== PYAUTOGUI TEST ===")
try:
    import pyautogui

    print("Moving mouse in 2 seconds... DO NOT TOUCH MOUSE")
    time.sleep(2)

    pyautogui.moveTo(500, 500, duration=0.5)
    pyautogui.click()
    print("[✓] Mouse move + click OK")

    pyautogui.typewrite("Hello from PyAutoGUI", interval=0.05)
    print("[✓] Typing OK")

except Exception as e:
    print("[✗] PyAutoGUI failed:", e)


# ── 3. SCREENSHOT TEST ───────────────────────────────────────────
print("\n=== SCREENSHOT TEST ===")
try:
    img = pyautogui.screenshot()
    img.save("test_screenshot.png")
    print("[✓] Screenshot saved as test_screenshot.png")
except Exception as e:
    print("[✗] Screenshot failed:", e)


# ── 4. GRID OVERLAY TEST (YOUR SYSTEM) ───────────────────────────
print("\n=== GRID OVERLAY TEST ===")
try:
    from computer import screenshot_grid

    b64, cells = screenshot_grid()
    import base64
    with open("grid_test.png", "wb") as f:
        f.write(base64.b64decode(b64))

    print(f"[✓] Grid screenshot saved (cells: {len(cells)})")
    print("Sample cells:", list(cells.keys())[:5])

except Exception as e:
    print("[✗] Grid system failed:", e)


# ── 5. PLAYWRIGHT TEST ───────────────────────────────────────────
print("\n=== PLAYWRIGHT TEST ===")
try:
    from computer import BrowserSession

    browser = BrowserSession()
    browser.start(headless=False, url="https://example.com")

    html = browser.get_html()
    print("[✓] Browser launched")
    print(f"[✓] HTML fetched ({len(html)} chars)")

    browser.close()

except Exception as e:
    print("[✗] Playwright failed:", e)


# ── 6. MODEL TEST (OPTIONAL) ─────────────────────────────────────
print("\n=== MODEL TEST ===")

provider = os.getenv("MODEL_PROVIDER", "none")
try:
    provider = os.getenv("MODEL_PROVIDER", "none")

    if provider == "gemini":
        from google import genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        client = genai.Client(api_key=api_key)

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Return ONLY this JSON: {\"msg\": \"hello\"}"
        )

        print("[✓] Gemini response:", resp.text[:100])

    elif provider == "claude":
        import os

        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key)

        resp = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=50,
            messages=[{"role": "user", "content": "Return ONLY JSON: {\"msg\":\"hello\"}"}]
        )

        print("[✓] Claude response:", resp.content[0].text[:100])

    elif provider == "openai":
        import os

        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Return ONLY JSON: {\"msg\":\"hello\"}"}]
        )

        print("[✓] OpenAI response:", resp.choices[0].message.content[:100])

    else:
        print("[!] Skipping model test (set MODEL_PROVIDER in .env)")

except Exception as e:
    print("[✗] Model test failed:", e)


print("\n=== DONE ===")