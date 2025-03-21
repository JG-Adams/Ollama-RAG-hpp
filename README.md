# RAGcpp

A lightweight Retrieval-Augmented Generation (RAG) system in C++ using Ollama-hpp for local language model inference and embedding-based retrieval.

## Features

Uses [Ollama-hpp](https://github.com/jmont-dev/ollama-hpp), a modern C++ wrapper for the Ollama API.

Implements vector-based text retrieval using cosine similarity.

Supports embedding generation and retrieval of relevant knowledge before LLM interaction.

Provides streaming chat responses for real-time AI interactions.

## Installation

### Requirements:

C++11 or later.

Ollama server running (sudo systemctl status ollama)
A compatible LLM and embedding model (e.g., llama3.2-latest, bge-base-en-v1.5-gguf)

### Prerequisite:

1. Install [Ollama](https://ollama.ai).
2. LLM `ollama pull llama3.2` (Or modify file for different model.)
3. A dedicated embedder. `ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf` Or any LLM with embedding ability.

### Setup:

1. Clone the repository:
2. Download Ollama-hpp from https://github.com/jmont-dev/ollama-hpp and place it in the project folder:
3. Compile the program:

## Run it:

### Usage:

1. Load a dataset:
Place a text file (cat-facts.txt) in the same directory. The program will embed and store the text for retrieval.

2. Ask a question:
When prompted, enter a query. The system will retrieve relevant information and generate a response using Ollama.

## License

This project is licensed under the MIT License—you are free to use, modify, and distribute it, but attribution is appreciated.

## Future Improvements

* Support for different embedding distance metrics (Euclidean, Manhattan, etc.)

* Multithreading for faster retrieval and generation

* Integration with a vector database (e.g., FAISS, SQLite)

This project is among the first C++ RAG implementations—contributions and feedback are welcome!
