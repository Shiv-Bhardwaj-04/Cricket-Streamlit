import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"Content of sys.path: {sys.path}")

try:
    import langchain
    print(f"Langchain version: {langchain.__version__}")
    print(f"Langchain file: {langchain.__file__}")
except ImportError as e:
    print(f"Failed to import langchain: {e}")

try:
    from langchain.chains import RetrievalQA
    print("Successfully imported RetrievalQA")
except ImportError as e:
    print(f"Failed to import RetrievalQA: {e}")

try:
    import langchain_community
    print(f"Langchain Community file: {langchain_community.__file__}")
except ImportError as e:
    print(f"Failed to import langchain_community: {e}")
