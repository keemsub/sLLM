!pip install -q langchain pypdf sentence-transformers chromadb openai
## Multi-Query Retriever

# Build a sample vectorDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load blog post
loader = WebBaseLoader("https://n.news.naver.com/mnews/article/003/0012317114?sid=105")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
splits = text_splitter.split_documents(data)

# VectorDB
model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

vectordb = Chroma.from_documents(documents=splits, embedding=ko_embedding)

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

question = "삼성전자 갤럭시 S24는 어떨 예정이야?"
llm = ChatOpenAI(temperature=0, openai_api_key = "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")
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectordb.as_retriever(), llm=llm
)

# Set logging for the queries
import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

unique_docs = retriever_from_llm.get_relevant_documents(query=question)
len(unique_docs)

unique_docs

## 기본 Parent-document Retriever

from langchain.retrievers import ParentDocumentRetriever

from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


loaders = [
    PyPDFLoader("/content/drive/MyDrive/강의 자료/[복지이슈 FOCUS 15ȣ] 경기도 극저신용대출심사모형 개발을 위한 국내 신용정보 활용가능성 탐색.pdf"),
    PyPDFLoader("/content/drive/MyDrive/강의 자료/[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load_and_split())

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

# This text splitter is used to create the child documents
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents", embedding_function=ko_embedding
)
# The storage layer for the parent documents
store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(docs, ids=None)

sub_docs = vectorstore.similarity_search("인공지능 예산")

print("글 길이: {}\n\n".format(len(sub_docs[0].page_content)))
print(sub_docs[0].page_content)

retrieved_docs = retriever.get_relevant_documents("인공지능 예산")

print("글 길이: {}\n\n".format(len(retrieved_docs[0].page_content)))
print(retrieved_docs[0].page_content)

## 본문의 Full_chunk가 너무 길때

# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="split_parents", embedding_function=ko_embedding
)
# The storage layer for the parent documents
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(docs)

len(list(store.yield_keys()))

sub_docs = vectorstore.similarity_search("인공지능 예산")

print(sub_docs[0].page_content)

len(sub_docs[0].page_content)

retrieved_docs = retriever.get_relevant_documents("인공지능 예산")

print(retrieved_docs[0].page_content)

len(retrieved_docs[0].page_content)

## Self-querying

pip install lark

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

docs = [
    Document(
        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
    ),
    Document(
        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
    ),
    Document(
        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
    ),
    Document(
        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
    ),
    Document(
        page_content="Toys come alive and have a blast doing so",
        metadata={"year": 1995, "genre": "animated"},
    ),
    Document(
        page_content="Three men walk into the Zone, three men walk out of the Zone",
        metadata={
            "year": 1979,
            "director": "Andrei Tarkovsky",
            "genre": "thriller",
            "rating": 9.9,
        },
    ),
]
vectorstore = Chroma.from_documents(docs, ko_embedding)

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chat_models import ChatOpenAI

metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie. One of ['science fiction', 'comedy', 'drama', 'thriller', 'romance', 'action', 'animated']",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The name of the movie director",
        type="string",
    ),
    AttributeInfo(
        name="rating", description="A 1-10 rating for the movie", type="float"
    ),
]
document_content_description = "Brief summary of a movie"
llm = ChatOpenAI(temperature=0, openai_api_key = "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")
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose = True
)

retriever.get_relevant_documents("what are some movies rated higher than 8.5")

## Time-weighted vector store Retriever

Scoring 방법 = *semantic_similarity + (1.0 - decay_rate) ^ hours_passed*

!pip install -q faiss-gpu

from datetime import datetime, timedelta

import faiss
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import Document
from langchain_community.vectorstores import FAISS


# Initialize the vectorstore as empty
embedding_size = 768
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(ko_embedding, index, InMemoryDocstore({}), {})
retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore, decay_rate=0.99, k=1
)

yesterday = datetime.now() - timedelta(days=1)
retriever.add_documents(
    [Document(page_content="영어는 훌륭합니다.", metadata={"last_accessed_at": yesterday})]
)
retriever.add_documents([Document(page_content="한국어는 훌륭합니다")])

# "Hello World" is returned first because it is most salient, and the decay rate is close to 0., meaning it's still recent enough
retriever.get_relevant_documents("영어가 좋아요")





## Ensemble Retriever

!pip install -q langchain pypdf sentence-transformers chromadb langchain-openai faiss-gpu --upgrade --quiet  rank_bm25 > /dev/null

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
encode_kwargs = {'normalize_embeddings': True}
ko_embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

loaders = [
    PyPDFLoader("/content/drive/MyDrive/강의 자료/[복지이슈 FOCUS 15ȣ] 경기도 극저신용대출심사모형 개발을 위한 국내 신용정보 활용가능성 탐색.pdf"),
    PyPDFLoader("/content/drive/MyDrive/강의 자료/[이슈리포트 2022-2호] 혁신성장 정책금융 동향.pdf"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load_and_split())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(docs)

# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 2



embedding = ko_embedding
faiss_vectorstore = FAISS.from_documents(texts, ko_embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

docs = ensemble_retriever.invoke("혁신정책금융과 극저신용대출모형의 차이")
for i in docs:

  print(i.metadata)
  print(":")
  print(i.page_content)
  print("-"*100)

faiss_vectorstore = FAISS.from_documents(texts, ko_embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 4})

docs = faiss_retriever.invoke("혁신정책금융과 극저신용대출모형의 차이")
for i in docs:

  print(i.metadata)
  print(":")
  print(i.page_content)
  print("-"*100)

import os
os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

openai = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0)

qa = RetrievalQA.from_chain_type(llm = openai,
                                 chain_type = "stuff",
                                 retriever = ensemble_retriever,
                                 return_source_documents = True)

query = "극저신용자 대출의 신용등급"
result = qa(query)
print(result['result'])

for i in result['source_documents']:
  print(i.metadata)
  print("-"*100)
  print(i.page_content)
  print("-"*100)

faiss_vectorstore = FAISS.from_documents(docs, ko_embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 4})

qa = RetrievalQA.from_chain_type(llm = openai,
                                 chain_type = "stuff",
                                 retriever = faiss_retriever,
                                 return_source_documents = True)

query = "극저신용자 대출의 신용등급"
result = qa(query)
print(result['result'])

for i in result['source_documents']:
  print(i.metadata)
  print("-"*100)
  print(i.page_content)
  print("-"*100)



## Long Context Reorder

from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI

texts = [
    "바스켓볼은 훌륭한 스포츠입니다.",
    "플라이 미 투 더 문은 제가 가장 좋아하는 노래 중 하나입니다.",
    "셀틱스는 제가 가장 좋아하는 팀입니다.",
    "보스턴 셀틱스에 관한 문서입니다.", "보스턴 셀틱스는 제가 가장 좋아하는 팀입니다.",
    "저는 영화 보러 가는 것을 좋아해요",
    "보스턴 셀틱스가 20점차로 이겼어요",
    "이것은 그냥 임의의 텍스트입니다.",
    "엘든 링은 지난 15 년 동안 최고의 게임 중 하나입니다.",
    "L. 코넷은 최고의 셀틱스 선수 중 한 명입니다.",
    "래리 버드는 상징적 인 NBA 선수였습니다.",
]

# Create a retriever
retriever = Chroma.from_texts(texts, embedding=ko_embedding).as_retriever(
    search_kwargs={"k": 10}
)
query = "셀틱스에 대해 어떤 이야기를 들려주시겠어요?"

# Get relevant documents ordered by relevance score
docs = retriever.get_relevant_documents(query)
docs

reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

# Confirm that the 4 relevant documents are at beginning and end.
reordered_docs

from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.prompts import PromptTemplate
import os
os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)

template = """Given this text extracts:
-----
{context}
-----
Please answer the following question:
{query}"""
prompt = PromptTemplate(
    template=template, input_variables=["context", "query"]
)
openai = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = 0)

llm_chain = LLMChain(llm=openai, prompt=prompt)
chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name="context"
)


reordered_result = chain.run(input_documents=reordered_docs, query=query)
result = chain.run(input_documents=docs, query=query)

print(reordered_result)
print("-"*100)
print(result)

