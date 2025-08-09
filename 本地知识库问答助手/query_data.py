import argparse
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function

CHROMA_PATH = "./chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
Use three sentences maximum and keep the answer concise.
"""

def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('query_text', type=str, help='The query text.')
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DataBase
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    print("📊 数据库统计:", db._collection.count())
    # 检查数据库内容
    collection_stas = db._collection.get(include=["embeddings"])
    if len(collection_stas['ids']) == 0:
        print('数据库为空，prepare_database有误')

    # Search the DataBase
    results = db.similarity_search_with_relevance_scores(query_text, k=5)
    print(f'检索结果数量：{len(results)}')
    for doc, score in results:
        print(f"Score:{score}, Source={doc.metadata.get('id')}, Preview={doc.page_content[:50]}")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model='qwen2.5:0.5b')
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response:{response_text}\nSources:{sources}"
    print(formatted_response)
    return response_text

if __name__ == '__main__':
    main()