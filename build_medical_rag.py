import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# =============================
# CONFIG
# =============================
BASE_URL = "https://medlineplus.gov/healthtopics.html"
OUTPUT_DIR = "vector_db"
USER_AGENT = "MedicalRAGBot/1.0 (educational use)"
REQUEST_DELAY = 1  # seconds

HEADERS = {"User-Agent": USER_AGENT}


# =============================
# STEP 1: GET ALL HEALTH TOPIC LINKS
# =============================
def get_topic_links():
    res = requests.get(BASE_URL, headers=HEADERS, timeout=30)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    links = set()
    banned = ["encyclopedia", "rss", "accessibility", "about", "news", "tools"]

    for a in soup.select("a[href]"):
        href = a.get("href")

        if not href:
            continue

        if (
            href.startswith("https://medlineplus.gov/")
            and href.endswith(".html")
            and not any(b in href.lower() for b in banned)
            and href.count("/") <= 4
        ):
            links.add(href)

    return sorted(list(links))


# =============================
# STEP 2: EXTRACT CLEAN CONTENT
# =============================
def extract_page_content(url):
    res = requests.get(url, headers=HEADERS, timeout=30)
    res.raise_for_status()

    soup = BeautifulSoup(res.text, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
        tag.decompose()

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown Topic"

    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if len(text) > 40:
            paragraphs.append(text)

    full_text = " ".join(paragraphs)
    return title, full_text


# =============================
# STEP 3: BUILD RAG DOCUMENTS
# =============================
def build_documents(topic_links):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    documents = []

    for url in tqdm(topic_links, desc="Processing pages"):
        try:
            title, text = extract_page_content(url)

            if len(text) < 300:
                continue

            chunks = splitter.split_text(text)

            for chunk in chunks:
                documents.append({
                    "content": chunk,
                    "metadata": {
                        "title": title,
                        "source": url,
                        "dataset": "MedlinePlus"
                    }
                })

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            print(f"⚠️ Skipped {url} | {e}")

    return documents


# =============================
# STEP 4: CREATE VECTOR DATABASE
# =============================
def build_vector_db(documents):
    print("\n🔹 Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [d["content"] for d in documents]
    metadatas = [d["metadata"] for d in documents]

    db = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db.save_local(OUTPUT_DIR)

    print(f"✅ Vector DB saved to: {OUTPUT_DIR}")
    return db


# =============================
# STEP 5: TEST RETRIEVAL
# =============================
def test_retrieval(db):
    query = "high fever for more than 3 days"
    print(f"\n🔍 Test Query: {query}\n")

    results = db.similarity_search(
        query,
        k=5,
        filter={"dataset": "MedlinePlus"}
    )

    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print("Title:", doc.metadata["title"])
        print("Source:", doc.metadata["source"])
        print(doc.page_content[:300], "...\n")


# =============================
# MAIN
# =============================
def main():
    print("🚀 Building Medical RAG Dataset (MedlinePlus)\n")

    topic_links = get_topic_links()
    print(f"📄 Found {len(topic_links)} health topic pages\n")

    documents = build_documents(topic_links)
    print(f"\n🧩 Total RAG chunks created: {len(documents)}")

    db = build_vector_db(documents)

    test_retrieval(db)

    print("🎉 DONE! Your medical RAG system is ready.")


if __name__ == "__main__":
    main()
