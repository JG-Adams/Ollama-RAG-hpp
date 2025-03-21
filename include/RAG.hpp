#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#include "ollama.hpp"

constexpr double SIMILARITY_THRESHOLD = 0.0; // Theoretical way to filter out useless finds. But the value is not consistent between different models.

inline void RAG_add_chunk_to_database(std::vector<std::pair<std::string, std::vector<float>>>& rag_database, const std::string& embedding_model, const std::string& chunk) {
    if (!chunk.empty()){
        ollama::response response = ollama::generate_embeddings(embedding_model, chunk);
        nlohmann::json json_response = response.as_json();
        std::vector<float> embedding = json_response["embeddings"][0].get<std::vector<float>>(); // Extract first array
        rag_database.emplace_back(chunk, embedding);
    }
}

inline float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot_product = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

inline std::vector<std::pair<std::string, float>> RAG_retrieve(std::vector<std::pair<std::string, std::vector<float>>>& rag_database, const std::string& embedding_model, const std::string& query, size_t top_n=3) {
    std::vector<std::pair<std::string, float>> similarities;
    if (!query.empty()){
    ollama::response response = ollama::generate_embeddings(embedding_model, query);
    nlohmann::json json_response = response.as_json();
    //if (json_response.is_array()){
        std::vector<float> query_embedding = json_response["embeddings"][0].get<std::vector<float>>(); // Extract first array

        for (const auto& [chunk, embedding] : rag_database) {
            float similarity = cosine_similarity(query_embedding, embedding);
            if (similarity > SIMILARITY_THRESHOLD){
                similarities.emplace_back(chunk, similarity);
            }
        }

        std::sort(similarities.begin(), similarities.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        if (similarities.size() > top_n) { similarities.resize(top_n); }
    }

    return similarities;
}

inline bool RAG_loadDocument(std::vector<std::pair<std::string, std::vector<float>>>& rag_database, const std::string& embedding_model, const std::string& path){
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Error: Could not open " + path + "\n";
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        RAG_add_chunk_to_database(rag_database, embedding_model, line);
    }
    file.close();
    return true;
}
