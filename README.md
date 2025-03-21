# RAGcpp

A lightweight Retrieval-Augmented Generation (RAG) system in C++ using Ollama-hpp for local language model inference and embedding-based retrieval.

## Features

Uses Ollama-hpp, a modern C++ wrapper for the Ollama API.

Implements vector-based text retrieval using cosine similarity.

Supports embedding generation and retrieval of relevant knowledge before LLM interaction.

Provides streaming chat responses for real-time AI interactions.

## Installation

Requirements:

C++11 or later

Ollama server running (sudo systemctl status ollama)

A compatible LLM and embedding model (e.g., llama3.2-latest, bge-base-en-v1.5-gguf)

## Setup

Clone the repository:

Download Ollama-hpp from https://github.com/jmont-dev/ollama-hpp and place it in the project folder:

Compile the program:

## Run it:

Usage:

Load a dataset

Place a text file (cat-facts.txt) in the same directory. The program will embed and store the text for retrieval.

Ask a question

When prompted, enter a query. The system will retrieve relevant information and generate a response using Ollama.

Example

License

This project is licensed under the MIT License—you are free to use, modify, and distribute it, but attribution is appreciated.

Future Improvements

Support for different embedding distance metrics (Euclidean, Manhattan, etc.)

Multithreading for faster retrieval and generation

Integration with a vector database (e.g., FAISS, SQLite)

This project is among the first C++ RAG implementations—contributions and feedback are welcome!
