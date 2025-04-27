from langchain_milvus import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os, warnings, logging

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

for name in (
    "",
    "langchain",
    "pymilvus",
    "httpx",
    "urllib3",
):
    logging.getLogger(name).setLevel(logging.CRITICAL)

os.environ["GLOG_minloglevel"] = "3"
os.environ["MILVUS_LOG_LEVEL"] = "fatal"

docs = TextLoader("DOG.md").load()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=".hf"
)

vecstore = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"uri": "./rag.db"},
    drop_old=True,
)

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="samples_1785",
    temperature=0.2,
    max_tokens=40,
)

rag = RetrievalQA.from_chain_type(llm, retriever=vecstore.as_retriever())

answer = rag.invoke("Can I use LLaMA 4 for Senior dog adoption programs?")

if isinstance(answer, dict):
    answer = answer["result"]
print("\n" + "=" * 50 + "\n" + answer.strip() + "\n" + "=" * 50)