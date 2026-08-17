from app.core.settings import GEMINI_API_KEY

if GEMINI_API_KEY:
    print("Gemini API key loaded: YES")
    print("Key length:", len(GEMINI_API_KEY))
else:
    print("Gemini API key loaded: NO")