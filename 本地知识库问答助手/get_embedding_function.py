from langchain_ollama import OllamaEmbeddings
from langchain_core.embeddings import Embeddings

# class EmbedWrapper(Embeddings):
#     def __init__(self, embed_model):
#         self.embed_model = embed_model

#     def embed_documents(self, texts):
#         vectors = []
#         for t in texts:
#             vec = self.embed_model.embed_query(t)
#             if vec is None:
#                 raise ValueError(f"Failed to embed document: {t}")
#             vectors.append(list(vec))
#         return vectors
    
#     def embed_query(self, text):
#         vec = self.embed_model.embed_query(text)
#         return list(vec) 


def get_embedding_function():
    embeddings = OllamaEmbeddings(model='bge-m3:latest')
    # test_vector = embeddings.embed_query("测试向量生成")
    # print(f"测试向量维度: {len(test_vector)}")
    # print(test_vector)
    return embeddings
    # return embeddings

# get_embedding_function()