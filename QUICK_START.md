# Quick Start Guide

## Your Sports Q&A Bot is Ready!

### The data is already processed and ready to use!

## To Run the App:

### Option 1: Use the batch file
```bash
run.bat
```

### Option 2: Run manually
```bash
streamlit run app/main.py
```

The app will open in your browser at http://localhost:8501

## How to Use:

1. The app loads automatically with pre-processed data
2. Simply type your question in the chat box
3. Get instant answers from the sports database!

## Example Questions:

- "Who is Virat Kohli?"
- "Tell me about UFC fighters"
- "What football matches are in the data?"
- "Show me cricket statistics"

## Note:

- Data is already indexed (35,709 chunks from your sports files)
- No need to process or reload data
- Just ask questions and get answers!
- Currently using document retrieval (install Ollama for AI-generated answers)

## If You Want AI-Generated Answers:

Install Ollama (free):
1. Download from https://ollama.com/download
2. Run: `ollama pull llama3.2`
3. Restart the app

Enjoy your Sports Q&A Bot! 🏆
