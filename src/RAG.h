#include <string>
#include <vector>

void RAG_add_chunk_to_database(std::vector<std::pair<std::string, std::vector<float>>>& rag_database, const std::string& embedding_model, const std::string& chunk);
std::vector<std::pair<std::string, float>> RAG_retrieve(std::vector<std::pair<std::string, std::vector<float>>>& rag_database, const std::string& embedding_model, const std::string& query, size_t top_n = 3);
bool RAG_loadDocument(std::vector<std::pair<std::string, std::vector<float>>>& rag_database, const std::string& embedding_model, const std::string& path);