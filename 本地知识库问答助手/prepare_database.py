import argparse
import os
import shutil
from uuid import uuid4
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores.chroma import Chroma
from langchain_ollama import OllamaEmbeddings

CHROMA_PATH = './chroma'
DATA_PATH = './data'

def load_documents():
    '''
    从指定目录加载PDF文档，每个页面作为一个文档
    '''
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"✅ 成功加载 {len(documents)} 个文档页面")
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # 每个块的最大字符数
        chunk_overlap=200, # 块之间的重叠字符数
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 文档分割完成，得到 {len(chunks)} 个文本块")

    # 为每个文本块添加唯一id并合并元数据
    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        page = chunk.metadata.get('page', 0)
        unique_id = f'{os.path.basename(source)}_page{page}_{str(uuid4())[:8]}'
        chunk.metadata['id'] = unique_id
        metadata_text = f"来源：{os.path.basename(source)}，页码：{page}\n"
        chunk.page_content = metadata_text + chunk.page_content

    # 🔍 检查前5个块的内容
    for i, chunk in enumerate(chunks[:5]):
        preview = chunk.page_content.strip().replace("\n", "\\n")
        print(f"📄 Chunk {i} 预览: {preview[:100]}")

    return chunks
    

def clear_database():
    '''
    清空数据库, 用于全量更新
    '''
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🗑 数据库已清空")


def main():
    clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embedding_function()
    # embeddings = OllamaEmbeddings(model='bge-m3:latest')

    # 🔍 检查嵌入是否正常
    try:
        test_vec = embeddings.embed_query("这是一个嵌入测试")
        print(f"🔍 嵌入测试：返回向量维度 {len(test_vec)}, 前5个值 {test_vec[:5]}")
    except Exception as e:
        print(f"❌ 嵌入生成失败: {e}")
        return
    
    try:
        test_vecs = embeddings.embed_documents([chunk.page_content for chunk in chunks[:3]])
        print(f"🔍 批量嵌入测试，返回向量数：{len(test_vecs)}，向量维度：{len(test_vecs[0])}")
    except Exception as e:
        print(f"❌ 批量嵌入失败: {e}")
        return

    # 🔍 检查前3个chunk的嵌入
    for i, chunk in enumerate(chunks[:3]):
        try:
            vec = embeddings.embed_query(chunk.page_content)
            print(f"📌 Chunk {i} 嵌入维度: {len(vec)}")
        except Exception as e:
            print(f"❌ Chunk {i} 嵌入失败: {e}")

    # ✅ 正式写入 Chroma
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"🎯 成功将 {len(chunks)} 个文本块添加到向量数据库")

if __name__ == '__main__':
    main()
