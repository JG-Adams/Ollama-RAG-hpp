#include <iostream>
#include <thread>
#include <atomic>

#include "RAG.hpp"

const std::string EMBEDDING_MODEL = "hf.co/CompendiumLabs/bge-base-en-v1.5-gguf";
const std::string LANGUAGE_MODEL = "llama3.2:latest";

// Each entry in the vector database is a pair of (chunk, embedding vector)
std::vector<std::pair<std::string, std::vector<float>>> RAG_DATABASE;
std::atomic_bool AI_busy = false;

int main() {
    RAG_loadDocument(RAG_DATABASE, EMBEDDING_MODEL, "cat-facts.txt");
    std::cout << "Loaded " << RAG_DATABASE.size() << " entries." << "\n";

    while(true){
        if (!AI_busy){
            std::cout << "Ask me a question (quit to exit): ";
            std::string input_query;
            std::getline(std::cin, input_query);
            if (input_query == "quit"){ break; }
            AI_busy = true;

            auto retrieved_knowledge = RAG_retrieve(RAG_DATABASE, EMBEDDING_MODEL, input_query);

            std::cout << "Retrieved knowledge:" << "\n";
            for (const auto& [chunk, similarity] : retrieved_knowledge) {
                std::cout << " - (similarity: " << similarity << ") " << chunk << "\n";
            }
            std::cout << "\nChatbot respond:\n";

            //std::string instruction_prompt = "You are a helpful chatbot.\nUse the following pieces of context provided by the embedder to answer questions if it's available without making up new information:\n";
            std::string instruction_prompt = "You are a helpful chatbot.\nKnowledge:\n";
            for (const auto& [chunk, _] : retrieved_knowledge) {
                instruction_prompt += " - " + chunk + "\n";
            }

            ollama::messages messages = {
                {"system", instruction_prompt},
                {"user", input_query}
            };

            ollama::request request(ollama::message_type::chat);
            request["model"]=LANGUAGE_MODEL;
            request["messages"]= messages.to_json();
            request["stream"] = true;
            ollama::chat(request, [](const ollama::response& response) {
                std::cout << response << std::flush;
                if (response.as_json()["done"]==true) {
                    AI_busy = false;
                }
            });
            std::cout << "\n";
        }else{
            continue;
        }
    }

    return 0;
}
