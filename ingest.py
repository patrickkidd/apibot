# from git import Repo
# REPO_URL = "https://github.com/owner/huge‑toolkit.git"
# LOCAL_PATH = "/tmp/huge‑toolkit"
# repo = Repo.clone_from(REPO_URL, LOCAL_PATH, depth=1)  # depth=1 ⇒ fast clone
# commit_sha = repo.head.commit.hexsha              # keep for provenance

import os
import pathlib, re
import shutil
import logging


LOCAL_PATH = "prompt_toolkit"
DB_PATH = "vector_db"

SRC_EXT = {".py", ".md", ".rst"}  # include notebooks if desired
EXCLUDE = re.compile(r"(^tests?/)|(_test\.py$)|(^docs?/build/)")

_log = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

_log.info(f"Scanning for {SRC_EXT} files in {LOCAL_PATH}.")
files = [
    p
    for p in pathlib.Path(LOCAL_PATH).rglob("*")
    if p.suffix in SRC_EXT and not EXCLUDE.search(str(p))
]


import ast, textwrap, hashlib, json


def split_code(path: pathlib.Path):
    src = path.read_text(encoding="utf‑8", errors="ignore")
    tree = ast.parse(src, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            loc = (node.lineno, node.end_lineno or node.lineno + 1)
            code = textwrap.dedent(
                "".join(src.splitlines(keepends=True)[loc[0] - 1 : loc[1]])
            )
            yield {
                "content": code,
                "metadata": {
                    "file": str(path.relative_to(LOCAL_PATH)),
                    "symbol": node.name,
                    "lineno": node.lineno,
                    # "commit": commit_sha
                },
            }


def split_markdown(path):
    paras = path.read_text("utf‑8").split("\n\n")
    for i, paragraph in enumerate(paras):
        if paragraph.strip():
            yield {
                "content": paragraph.strip(),
                "metadata": {
                    "file": str(path.relative_to(LOCAL_PATH)),
                    "chunk": i,
                    # "commit": commit_sha
                },
            }


_log.info(f"Reading and chunking {len(files)} files.")
chunks = []
for path in files:
    if path.suffix == ".py":
        chunks.extend(split_code(path))
    else:
        chunks.extend(split_markdown(path))
_log.info(f"{len(chunks):,} chunks ready")


from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document

documents = [
    Document(page_content=d["content"], metadata=d["metadata"]) for d in chunks
]

if os.path.isdir(DB_PATH):
    shutil.rmtree(DB_PATH)

_log.info("Creating vector store")

emb = OpenAIEmbeddings(model="text-embedding-3-small")  # or your own
store = Chroma.from_documents(
    documents=documents,
    embedding=emb,
    persist_directory=DB_PATH,
)
