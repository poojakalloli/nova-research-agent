# NOVA - Research & Calculator Agent

A Streamlit app built with LangChain. It can search the web (Tavily), calculate, find YouTube videos,
check the weather, and open Gmail / Google Calendar. If one AI provider fails
(Gemini, Groq, OpenRouter, Hugging Face) it automatically switches to the next one.

## Features
- Web search with Tavily (falls back to Serper, then DuckDuckGo)
- Safe arithmetic: add, subtract, multiply, divide and more (no `eval`)
- Weather and 3-day forecast (Open-Meteo, no API key needed)
- YouTube search links, Gmail search/draft, Google Calendar event links
- Automatic AI provider fallback and auto-refresh when you switch tools

## Setup (Windows, VS Code terminal)
```
pip install -r requirements.txt
copy .env.example .env
```
Open `.env` and add your own API keys, then run:
```
streamlit run app.py
```

## Tech
Python, Streamlit, LangChain, Gemini / Groq / OpenRouter / Hugging Face, Tavily, Open-Meteo
