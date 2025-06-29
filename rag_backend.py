import json
import os
import sqlite3
import requests
from datetime import datetime
from pathlib import Path
import trafilatura
from trafilatura.settings import use_config
import faiss
import numpy as np
import requests
import nbformat
from nbconvert import MarkdownExporter
import time

import ollama
from urllib.parse import urlparse, urljoin
import hashlib
import re
from typing import List, Dict, Tuple
import logging
from markitdown import MarkItDown
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from tqdm import tqdm
from PIL import Image
import shutil
import io
import base64
import sys
import pymupdf4llm
import socket

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

host_node = socket.gethostname()
base_url=f"http://localhost:11435/"

EMBED_URL = f"{base_url}api/embeddings"
OLLAMA_CHAT_URL = f"{base_url}api/chat"
OLLAMA_URL = f"{base_url}api/generate"
EMBED_MODEL = "nomic-embed-text"
GEMMA_MODEL = "gemma3:12b"
PHI_MODEL = "phi4:latest"
CHUNK_SIZE = 256
CHUNK_OVERLAP = 40
MAX_CHUNK_LENGTH = 512  # characters
TOP_K = 3  # FAISS top-K matches
ROOT = Path(os.getcwd())


class Doc_Processor:
    def __init__(self):
        # Trafilatura config
        self.config = use_config()
        self.config.set("DEFAULT", "EXTRACTION_TIMEOUT", "30")
    
    def log(self, level: str, message: str) -> None:
        sys.stderr.write(f"{level}: {message}\n")
        sys.stderr.flush()
            
        
    def get_embedding(self, text: str) -> np.ndarray:
        response = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
        response.raise_for_status()
        return np.array(response.json()["embedding"], dtype=np.float32)
    
    def caption_image(self, img_url_or_path: str) -> str:
        self.log("CAPTION", f"🖼️ Attempting to caption image: {img_url_or_path}")

        full_path = Path(__file__).parent / "documents" / img_url_or_path
        full_path = full_path.resolve()

        if not full_path.exists():
            self.log("ERROR", f"❌ Image file not found: {full_path}")
            return f"[Image file not found: {img_url_or_path}]"

        try:
            if img_url_or_path.startswith("http"): # for extract_web_pages
                response = requests.get(img_url_or_path)
                encoded_image = base64.b64encode(response.content).decode("utf-8")
            else:
                with open(full_path, "rb") as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

            # Set stream=True to get the full generator-style output
            with requests.post(OLLAMA_URL, json={
                "model": GEMMA_MODEL,
                "prompt": "If there is lot of text in the image, then ONLY reply back with exact text in the image, else Describe the image such that your response can replace 'alt-text' for it. Only explain the contents of the image and provide no further explaination.",
                "images": [encoded_image],
                "stream": True
            }, stream=True) as response:

                caption_parts = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        caption_parts.append(data.get("response", ""))
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue  # silently skip malformed lines

                caption = "".join(caption_parts).strip()
                self.log("CAPTION", f"✅ Caption generated: {caption}")
                return caption if caption else "[No caption returned]"

        except Exception as e:
            self.log("ERROR", f"⚠️ Failed to caption image {img_url_or_path}: {e}")
            return f"[Image could not be processed: {img_url_or_path}]"
    
    def replace_images_with_captions(self, markdown: str) -> str:
        def replace(match):
            alt, src = match.group(1), match.group(2)
            try:
                caption = self.caption_image(src)
                # Attempt to delete only if local and file exists
                if not src.startswith("http"):
                    img_path = Path(__file__).parent / "documents" / src
                    if img_path.exists():
                        img_path.unlink()
                        self.log("INFO", f"🗑️ Deleted image after captioning: {img_path}")
                return f"**Image:** {caption}"
            except Exception as e:
                self.log("WARN", f"Image deletion failed: {e}")
                return f"[Image could not be processed: {src}]"

        return re.sub(r'!\[(.*?)\]\((.*?)\)', replace, markdown)


    def download_ipynb_files_from_github_api(self, repo, folder_path, branch="main", dest_folder="downloaded_notebooks"):
        """
        Download all .ipynb files from a GitHub repo folder using the GitHub API.

        Args:
            repo (str): e.g., "NVIDIA/accelerated-computing-hub"
            folder_path (str): e.g., "gpu-python-tutorial"
            branch (str): usually "main" or "master"
            dest_folder (str): local directory to store downloaded notebooks
        """
        
        final_dest_folder = ROOT / "documents" / dest_folder

        api_url = f"https://api.github.com/repos/{repo}/contents/{folder_path}?ref={branch}"
        headers = {"Accept": "application/vnd.github.v3+json"}
        response = requests.get(api_url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to access GitHub API: {response.status_code} - {response.text}")
            return

        os.makedirs(final_dest_folder, exist_ok=True)
        files = response.json()
        found = False

        for file in files:
            if file["name"].endswith(".ipynb"):
                download_url = file["download_url"]
                local_path = os.path.join(final_dest_folder, file["name"])
                r = requests.get(download_url)
                with open(local_path, "wb") as f:
                    f.write(r.content)
                print(f"Downloaded: {file['name']}")
                found = True

        if not found:
            print("No .ipynb files found in the folder.")

        return
   
    def convert_notebooks_to_markdown(self, notebook_folder, markdown_folder):
        
        """Convert .ipynb files to markdown"""

        markdown_final_folder = ROOT / "documents" / markdown_folder
        nb_folder_final = ROOT / "documents" / notebook_folder

        os.makedirs(markdown_final_folder, exist_ok=True)

        for filename in os.listdir(nb_folder_final):
            if filename.endswith(".ipynb"):
                notebook_path = os.path.join(nb_folder_final, filename)
                with open(notebook_path, "r", encoding="utf-8") as f:
                    nb_node = nbformat.read(f, as_version=4)

                exporter = MarkdownExporter()
                body, _ = exporter.from_notebook_node(nb_node)

                md_filename = os.path.splitext(filename)[0] + ".md"
                md_path = os.path.join(markdown_final_folder, md_filename)
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(body)

                print(f"Converted: {filename} → {md_filename}")

        return

    def extract_webpage(self, url):
        """Extract and convert webpage content to markdown. Usage: extract_webpage|input={"url": "https://example.com"}"""

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "Failed to download the webpage."

        markdown = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            include_images=True,
            output_format='markdown'
        ) or ""

        markdown = self.replace_images_with_captions(markdown)
        return markdown
    
    def extract_pdf(self, file_path: str):
        """Convert PDF file content to markdown format. Usage: extract_pdf|input={"file_path": "documents/dlf.pdf"}"""

        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        ROOT = Path(__file__).parent.resolve()
        global_image_dir = ROOT / "documents" / "images"
        global_image_dir.mkdir(parents=True, exist_ok=True)

        # Actual markdown with relative image paths
        markdown = pymupdf4llm.to_markdown(
            file_path,
            write_images=True,
            image_path=str(global_image_dir)
        )

        # Re-point image links in the markdown
        markdown = re.sub(
            r'!\[\]\((.*?/images/)([^)]+)\)',
            r'![](images/\2)',
            markdown.replace("\\", "/")
        )

        markdown = self.replace_images_with_captions(markdown)
        return markdown

    
    def semantic_merge(self, text: str) -> list[str]:
        """Splits text semantically using LLM: detects second topic and reuses leftover intelligently."""
        WORD_LIMIT = 512
        words = text.split()
        i = 0
        final_chunks = []

        while i < len(words):
            # 1. Take next chunk of words (and prepend leftovers if any)
            chunk_words = words[i:i + WORD_LIMIT]
            chunk_text = " ".join(chunk_words).strip()

            prompt = f"""
    You are a markdown document segmenter.

    Here is a portion of a markdown document:

    ---
    {chunk_text}
    ---

    If this chunk clearly contains **more than one distinct topic or section**, reply ONLY with the **second part**, starting from the first sentence or heading of the new topic.

    If it's only one topic, reply with NOTHING.

    Keep markdown formatting intact.
    """

            try:
                response = requests.post(OLLAMA_CHAT_URL, json={
                    "model": PHI_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                })
                reply = response.json().get("message", {}).get("content", "").strip()

                if reply:
                    # If LLM returned second part, separate it
                    split_point = chunk_text.find(reply)
                    if split_point != -1:
                        first_part = chunk_text[:split_point].strip()
                        second_part = reply.strip()

                        final_chunks.append(first_part)

                        # Get remaining words from second_part and re-use them in next batch
                        leftover_words = second_part.split()
                        words = leftover_words + words[i + WORD_LIMIT:]
                        i = 0  # restart loop with leftover + remaining
                        continue
                    else:
                        # fallback: if split point not found
                        final_chunks.append(chunk_text)
                else:
                    final_chunks.append(chunk_text)

            except Exception as e:
                self.log("ERROR", f"Semantic chunking LLM error: {e}")
                final_chunks.append(chunk_text)

            i += WORD_LIMIT

        return final_chunks

    
    def process_documents(self, process_urls=False):
        """Process documents and create FAISS index using unified multimodal strategy."""
        self.log("INFO", "Indexing documents with unified RAG pipeline...")
        ROOT = Path(os.getcwd())
        DOC_PATH = ROOT / "documents"
        INDEX_CACHE = ROOT / "faiss_index"
        INDEX_CACHE.mkdir(exist_ok=True)
        INDEX_FILE = INDEX_CACHE / "index.bin"
        METADATA_FILE = INDEX_CACHE / "metadata.json"
        CACHE_FILE = INDEX_CACHE / "doc_index_cache.json"
        processing_times = []
        mkdown_details = {}
        
        if process_urls:
            process_github_start = time.time()
            mkdown_folder_path = ROOT / "documents" / "converted_markdown" 
            
            _ = self.download_ipynb_files_from_github_api(
                                        repo="NVIDIA/accelerated-computing-hub",
                                        folder_path="gpu-python-tutorial",
                                        branch="main"
                                    ),
            
            _ = self.convert_notebooks_to_markdown(
                                        notebook_folder="downloaded_notebooks",
                                        markdown_folder="converted_markdown"
            )


            for markdown_file in os.listdir(mkdown_folder_path):
                if markdown_file.endswith(".md"):
                    with open(os.path.join(mkdown_folder_path, markdown_file), "r", encoding="utf-8") as f:
                            mk_text= f.read()
                    
                    mk_text = re.sub(
                                        r'!\[\]\((.*?/images/)([^)]+)\)',
                                        r'![](images/\2)',
                                        mk_text.replace("\\", "/")
                                )

                    #mk_text = self.replace_images_with_captions(mk_text)
                    mkdown_details[os.path.join(mkdown_folder_path, markdown_file)] = mk_text 

            process_github_end = time.time()
            
            processing_times.append(f"Time taken to download and convert ipynb to markdown: {process_github_end - process_github_start}")


        def file_hash(path):
            return hashlib.md5(Path(path).read_bytes()).hexdigest()

        creating_faiss_gpu_start = time.time()

        CACHE_META = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        metadata = json.loads(METADATA_FILE.read_text()) if METADATA_FILE.exists() else []
        index = faiss.read_index(str(INDEX_FILE)) if INDEX_FILE.exists() else None

        for file in DOC_PATH.glob("*.*"):
            fhash = file_hash(file)
            if file.name in CACHE_META and CACHE_META[file.name] == fhash:
                self.log("SKIP", f"Skipping unchanged file: {file.name}")
                continue

            self.log("PROC", f"Processing: {file.name}")
            try:
                ext = file.suffix.lower()
                markdown = ""

                if ext == ".pdf":
                    self.log("INFO", f"Using MuPDF4LLM to extract {file.name}")
                    markdown = self.extract_pdf(file_path=str(file))

                elif ext in [".html", ".htm", ".url"]:
                    self.log("INFO", f"Using Trafilatura to extract {file.name}")
                    markdown = self.extract_webpage(url=file.read_text().strip())

                else:
                    # Fallback to MarkItDown for other formats
                    converter = MarkItDown()
                    self.log("INFO", f"Using MarkItDown fallback for {file.name}")
                    markdown = converter.convert(str(file)).text_content

                if not markdown.strip():
                    self.log("WARN", f"No content extracted from {file.name}")
                    continue

                if len(markdown.split()) < 10:
                    self.log("WARN", f"Content too short for semantic merge in {file.name} → Skipping chunking.")
                    chunks = [markdown.strip()]
                else:
                    self.log("INFO", f"Running semantic merge on {file.name} with {len(markdown.split())} words")
                    chunks = self.semantic_merge(markdown)


                embeddings_for_file = []
                new_metadata = []
                for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {file.name}")):
                    embedding = self.get_embedding(chunk)
                    embeddings_for_file.append(embedding)
                    new_metadata.append({
                        "doc": file.name,
                        "chunk": chunk,
                        "chunk_id": f"{file.stem}_{i}"
                    })

                if embeddings_for_file:
                    if index is None:
                        dim = len(embeddings_for_file[0])
                        index = faiss.IndexFlatL2(dim)
                    index.add(np.stack(embeddings_for_file))
                    metadata.extend(new_metadata)
                    CACHE_META[file.name] = fhash

                    # ✅ Immediately save index and metadata
                    CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
                    METADATA_FILE.write_text(json.dumps(metadata, indent=2))
                    faiss.write_index(index, str(INDEX_FILE))
                    self.log("SAVE", f"Saved FAISS index and metadata after processing {file.name}")

            except Exception as e:
                self.log("ERROR", f"Failed to process {file.name}: {e}")

        if mkdown_details:
            
            self.log("PROC", "Processing Markdown Files")

            for mkfile in mkdown_details:
                fhash = file_hash(mkfile)
                markdown = mkdown_details[mkfile]
                chunks = self.semantic_merge(markdown)

                embeddings_for_file = []
                new_metadata = []
                
                for i, chunk in enumerate(tqdm(chunks, desc=f"Embedding {mkfile}")):
                    embedding = self.get_embedding(chunk)
                    embeddings_for_file.append(embedding)
                    new_metadata.append({
                        "doc": mkfile,
                        "chunk": chunk,
                        "chunk_id": f"{mkfile}_{i}"
                    })

                if embeddings_for_file:
                    if index is None:
                        dim = len(embeddings_for_file[0])
                        index = faiss.IndexFlatL2(dim)
                    index.add(np.stack(embeddings_for_file))
                    metadata.extend(new_metadata)
                    CACHE_META[mkfile] = fhash

                    # ✅ Immediately save index and metadata
                    CACHE_FILE.write_text(json.dumps(CACHE_META, indent=2))
                    METADATA_FILE.write_text(json.dumps(metadata, indent=2))
                    faiss.write_index(index, str(INDEX_FILE))
                    self.log("SAVE", f"Saved FAISS index and metadata after processing markdown file {mkfile}")
        
        creating_faiss_gpu_end = time.time()
        processing_times.append(f"Time taken to create FAISS Index: {creating_faiss_gpu_end - creating_faiss_gpu_start}")

        return processing_times


    def ensure_faiss_ready(self,):
        from pathlib import Path
        index_path = ROOT / "faiss_index" / "index.bin"
        meta_path = ROOT / "faiss_index" / "metadata.json"
        if not (index_path.exists() and meta_path.exists()):
            self.log("INFO", "Index not found — running process_documents()...")
            self.process_documents()
        else:
            self.log("INFO", "Index already exists. Skipping regeneration.")

    def search_documents(self, query: str, k: int) -> list[str]:
        """Search indexed documents for relevant content. Usage: search_documents|query="india Current GDP" """
        self.ensure_faiss_ready()
        self.log("SEARCH", f"Query: {query}")
        try:
            index = faiss.read_index(str(ROOT / "faiss_index" / "index.bin"))
            metadata = json.loads((ROOT / "faiss_index" / "metadata.json").read_text())
            query_vec = self.get_embedding(query).reshape(1, -1)
            D, I = index.search(query_vec, k=k)
            results = []
            for idx in I[0]:
                data = metadata[idx]
                results.append(f"{data['chunk']}\n[Source: {data['doc']}, ID: {data['chunk_id']}]")
            return results
        except Exception as e:
            return [f"ERROR: Failed to search: {str(e)}"]   


    def run_job(self):
        
        """Run the processing job"""
        
        print("Start URL processing")
        
        processing_times = self.process_documents(process_urls=True)

        with open("time_log.log", "w") as f:
            for line in processing_times:
                f.write(line + "\n")
        
        return


if __name__ == "__main__":
    url_parser = Doc_Processor()
    url_parser.run_job() 