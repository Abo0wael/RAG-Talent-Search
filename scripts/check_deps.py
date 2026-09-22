import sys

pkgs = [
    'langchain',
    'langchain_community',
    'chromadb',
    'sentence_transformers',
    'streamlit',
    'langchain_groq',
    'dotenv',
    'pandas',
    'matplotlib',
    'plotly'
]

missing = []
for p in pkgs:
    try:
        __import__(p)
        print(f"OK: {p}")
    except Exception as e:
        print(f"MISSING: {p} ({e})")
        missing.append(p)

print(f"\nTotal missing: {len(missing)}")
if missing:
    sys.exit(1)
