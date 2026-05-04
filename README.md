# Desk Pilot

AI agent for controlling a browser and basic desktop interactions using LLMs (Gemini / Claude / OpenAI).


## SETUP

Install dependencies:

```bash
pip install -e .
playwright install
```

Put your keys in a .env file
```
MODEL_PROVIDER=gemini
GOOGLE_API_KEY=API-KEY-HERE
```

Run a test with test.py to see if all works(you may need to change the models and stuffs)

## USAGE

Run with a URL and a goal:

```
python main.py --url "https://example.com
" --goal "Click any link"
```

Example:
```
python main.py --url "https://vespertil.com
" --goal "Go to the contact section"
```
## MODES

HTML mode (default)
Uses Playwright. More reliable. Prefer this for web tasks.

Screenshot mode
Uses PyAutoGUI. Required for desktop control, but less stable.

Example:

```
python main.py --mode screenshot --goal "Open Chrome and type google.com"
```

## KEY NOTES


Browser is launched maximized
Required for correct click alignment
Do not use mouse/keyboard while the agent is running
Screenshot mode should only be used when necessary

## CURRENT LIMITATIONS

No action verification (agent does not confirm if actions worked)
May switch to screenshot mode unnecessarily
Can misclick or loop on complex tasks
No window or focus awareness

## CODE STRUCTURE

main.py → CLI entry point
agent.py → agent logic and model interaction
computer.py → browser and mouse control

## NEXT IMPROVEMENTS

Add action verification
Add retry logic
Restrict click regions (avoid OS UI like minimize/close)
Improve prompting and mode selection

## STATUS

Works for simple tasks and demos.
Needs improvement for reliability in multi-step workflows