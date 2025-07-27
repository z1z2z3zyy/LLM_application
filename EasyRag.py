from typing import List
from sentence_transformers import SentenceTransformer,CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb

def split_doc(doc: str) -> List[str]:
    with open(doc,"r",encoding="utf-8") as f:
        text = f.read()
    return [chunk for chunk in text.split("\n\n")]

def embed_chunk(chunk:str,model)->List[float]:
    embedding = model.encode(chunk, normalize_embeddings=True)
    return embedding

def save_embedding(chunks:List[str],embeddings:List[float],collection) -> None:
    for i,chunk,embedding in zip(range(len(chunks)),chunks,embeddings):
        collection.add(documents=[chunk],embeddings=[embedding],ids=[str(i)])

def retrieve(query:str,topk:int,collection,model)->List[str]:
    embedding_query = embed_chunk(query,model)
    results=collection.query(query_embeddings=[embedding_query],n_results=topk)
    return results['documents'][0]

def rerank(query:str,retrieve_results:List[str])->List[str]:
    cross_encoder=CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    pairs=[(query,chunk) for chunk in retrieve_results]
    scores=cross_encoder.predict(pairs)
    scored_chunks=list(zip(retrieve_results,scores))
    scored_chunks.sort(key=lambda x:x[1],reverse=True)
    return [chunk for chunk,_ in scored_chunks]

def generate(query:str,informations:List[str]):
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。相关信息:{informations}
           ,请基于以上信息回答，不要编造内容，如果相关信息中没有和用户问题相关内容，就回复“不知道”"""},
            {"role": "user", "content": f"{query}"},
        ],

    )
    content = completion.choices[0].message.content
    print(content)

if __name__ == "__main__":
    load_dotenv()
    chunks = split_doc("resources/doc.md")
    embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    embeddings = [embed_chunk(chunk,embedding_model) for chunk in chunks]
    chromadb_client = chromadb.EphemeralClient()
    chromadb_collection = chromadb_client.get_or_create_collection(name="default")
    save_embedding(chunks,embeddings,chromadb_collection)
    query="是什么东西让世界静止了"
    retrieve_txt=retrieve(query,3,chromadb_collection,embedding_model)
    # for i,txt in enumerate(retrieve_txt):
    #     print(i,txt)
    final_results=rerank(query,retrieve_txt)
    # for i,txt in enumerate(final_results):
    #     print(i,txt)
    generate(query,final_results)