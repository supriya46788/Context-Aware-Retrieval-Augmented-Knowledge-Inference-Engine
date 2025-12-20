# Context-Aware Retrieval-Augmented Knowledge Inference Engine

A modern **Retrieval-Augmented Generation (RAG)** application built using **Google Gemini**, **LangChain**, **ChromaDB**, and **Gradio**.

The system answers user questions **strictly based on locally provided documents**. It is designed to overcome fundamental limitations of Large Language Models (LLMs), specifically **hallucination** and **lack of updated information beyond their knowledge cutoff**.

---

## 🚨 Core Problems with Large Language Models

### Problem 1: Hallucination

Large Language Models are probabilistic text generators. As a result, they often:

* Produce confident but factually incorrect responses
* Fabricate information when relevant knowledge is missing
* Do not clearly state uncertainty

This behavior makes LLMs unsuitable for enterprise, research, and decision‑critical applications.

---

### Problem 2: Static and Non‑Contextual Knowledge

Large Language Models:

* Are trained on static, historical datasets
* Cannot access private, internal, or domain‑specific documents
* Cannot adapt to new or frequently updated information

This severely limits their usability in real‑world systems that require dynamic and context‑aware intelligence.

---

## ✅ How This Project Solves These Problems

* Uses relevant document content before answering, so the model does not guess
* Fetches fresh information from local files instead of relying on old training data
* Clearly responds with “I don’t know” when the answer is not found in documents
* Breaks documents into smaller chunks for faster and more accurate retrieval
* Separates ingestion, retrieval, and inference for easy updates and scalability

---

## 🧠 How It Works (RAG Flow)

1. Documents are loaded from the `data/` folder
2. Documents are split into small chunks
3. Each chunk is converted into vector embeddings
4. Embeddings are stored in ChromaDB
5. User queries retrieve the most relevant chunks
6. Gemini generates answers using retrieved context only

---

## 🛠️ Tech Stack Used

* **Programming Language:** Python
* **Large Language Model:** Google Gemini
* **Framework:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** Google Generative AI Embeddings
* **Frontend / UI:** Gradio
* **Environment Management:** Python `venv`, `dotenv`

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/context-aware-rag-engine.git
cd context-aware-rag-engine
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```text
GOOGLE_API_KEY=your_google_gemini_api_key
```

### 5️⃣ Ingest Documents

```bash
python ingest.py
```

> Any update to the `data/` directory requires re‑running the ingestion step.

### 6️⃣ Run the Application

```bash
python app.py
```

Access the application at:

```
http://localhost:7860
```
