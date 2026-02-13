import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DATA_PATH = "data/cricket"

def load_documents():
    """Load all cricket CSV files and convert to documents"""
    documents = []
    
    categories = ["Batting", "Bowling", "Fielding"]
    formats = {
        "Batting": ["ODI data.csv", "t20.csv", "test.csv"],
        "Bowling": ["Bowling_ODI.csv", "Bowling_t20.csv", "Bowling_test.csv"],
        "Fielding": ["Fielding_ODI.csv", "Fielding_t20.csv", "Fielding_test.csv"]
    }
    
    for category in categories:
        path = os.path.join(DATA_PATH, category)
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue
            
        print(f"Loading {category} data...")
        
        for file_name in formats[category]:
            file_path = os.path.join(path, file_name)
            if not os.path.exists(file_path):
                print(f"  Warning: {file_name} not found")
                continue
            
            # Determine format
            if 'ODI' in file_name or 'ODI' in file_name:
                format_type = 'ODI'
            elif 't20' in file_name.lower():
                format_type = 'T20'
            elif 'test' in file_name.lower():
                format_type = 'Test'
            else:
                format_type = 'Unknown'
            
            # Load CSV
            df = pd.read_csv(file_path)
            
            # Convert each row to a document
            for idx, row in df.iterrows():
                # Create readable text from row
                text_parts = []
                player_name = row.get('Player', 'Unknown Player')
                
                text_parts.append(f"Player: {player_name}")
                text_parts.append(f"Format: {format_type}")
                text_parts.append(f"Category: {category}")
                
                # Add all stats
                for col, val in row.items():
                    if col not in ['Player', 'Unnamed: 0'] and pd.notna(val):
                        text_parts.append(f"{col}: {val}")
                
                text = " | ".join(text_parts)
                
                # Create document
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': file_path,
                        'player': player_name,
                        'format': format_type,
                        'category': category,
                        'row': idx
                    }
                )
                documents.append(doc)
            
            print(f"  [OK] Loaded {len(df)} records from {file_name}")
    
    print(f"\nTotal documents loaded: {len(documents)}")
    return documents

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    docs = load_documents()
    if docs:
        chunks = split_documents(docs)
        print(f"\n[OK] Ready to index {len(chunks)} chunks")
    else:
        print("No documents found!")
