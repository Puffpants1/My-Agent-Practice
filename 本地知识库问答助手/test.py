from langchain_community.vectorstores.chroma import Chroma
from get_embedding_function import get_embedding_function


db = Chroma(
    persist_directory="./chroma",
    embedding_function=get_embedding_function()
)

stored = db._collection.get(include=["embeddings"])
# print(stored)
print(f"数据库内文档数: {len(stored['ids'])}")
print(f"第一条向量维度: {stored['embeddings'][0]}")

# embeddings = get_embedding_function()
# test_texts = ["这是第一个文本", "这是第二个文本"]
# try:
#     vecs = embeddings.embed_documents(test_texts)
#     print(f"批量嵌入返回 {len(vecs)} 条向量，每条维度 {len(vecs[0])}")
# except Exception as e:
#     print("批量嵌入失败:", e)

print("📊 数据库统计:", db._collection.count())
print("🔍 测试检索:", db.similarity_search("测试", k=3))