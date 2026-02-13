# Sports Q&A Bot

A Streamlit-based chatbot that answers questions about Football, Cricket, and UFC using RAG (Retrieval Augmented Generation).

## Setup Instructions

### 1. Install Dependencies
Run the setup script:
```bash
setup.bat
```

Or manually install:
```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key
Edit the `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### 3. Process Data (Optional - First Time)
If you want to pre-process the data before running the app:
```bash
python train_and_test.py
```

### 4. Run the Streamlit App
Run the app script:
```bash
run.bat
```

Or manually:
```bash
streamlit run app/main.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. Click "Process/Reload Data" in the sidebar to load and index your sports data
2. Ask questions in the chat interface
3. View sources for each answer

## Data Structure

Place your data files in:
- `data/football/` - Football-related CSV/PDF/TXT files
- `data/cricket/` - Cricket-related CSV/PDF/TXT files
- `data/ufc/` - UFC-related CSV/PDF/TXT files
