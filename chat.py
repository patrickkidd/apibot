import logging

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

_log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

DB_PATH = "vector_db"

PROMPT_TEMPLATE = """
You are an expert on source code experts listed below. Answer the following
developer's question using ONLY the source code excerpts as context. If unsure,
say you don't know.

DEVELOPER'S QUESTION:

{question}

---------------------------------------------

SOURCE CODE EXERPTS AS CONTEXT:

{context}
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

emb = OpenAIEmbeddings(model="text-embedding-3-small")


_log.info(f"Opening vector store at {DB_PATH}")
store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=emb,
)


qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini"),
    chain_type="stuff",
    return_source_documents=True,
    retriever=store.as_retriever(search_kwargs=dict(k=20)),
    chain_type_kwargs=dict(prompt=prompt.partial()),
)


while True:
    question = input("Question: ")
    if not question.strip():
        continue
    answer = qa.invoke(question)
    result = answer["result"]
    print(result)
    print("----------------------------------------")
    print("SOURCE DOCUMENTS:")
    print("----------------------------------------")
    for document in answer["source_documents"]:
        print(f"FILE: {document.metadata['file']}:{document.metadata['lineno']}")
        print(" ")
        print(document.page_content)
        print("----------------------------------------")
