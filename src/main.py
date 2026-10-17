import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
chat = client.chats.create(
    model="gemini-3.5-flash-lite",
    config=types.GenerateContentConfig(temperature=0),
)

while True:
    msg = input("you> ").strip()
    if msg in {"exit", "quit"}:
        break
    try:
        print("bot>", chat.send_message(msg).text)
    except Exception as e:
        print("error>", e)
