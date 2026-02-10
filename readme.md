## RAG_MVP (Using LangChain)

A simple project to learn **RAG (Retrieval-Augmented Generation)** chunking techniques using LangChain.

- **What is RAG?** → Feed your own documents to an LLM so it answers from your data, not from its training.
- **What is Chunking?** → Breaking large text into smaller pieces so the LLM can search and find relevant parts.

---

### Project Structure

```
rag_mvp/
├── chunk-lab/                          ← Chunking experiments (run one by one)
│   ├── fixed_size.py                   ← Cut every N characters
│   ├── recursive_char_text_split.py    ← Cut every N chars + overlap
│   ├── token_based.py                  ← Cut every N tokens
│   ├── similarity_based.py             ← Cut where meaning changes
│   └── document-based/
│       ├── header_based.py             ← Cut by markdown headers
│       └── pdf_based.py                ← Cut by PDF structure
│
├── app/
│   └── ask.py                          ← Ask questions → get answers from LLM
│
├── data/
│   └── company_policy.txt              ← Sample document
│
├── vector_db/                          ← Auto-generated (not committed)
├── requirements.txt
├── .gitignore
└── readme.md
```

---

### Quick Start

```bash
# 1. Clone
git clone https://github.com/your-username/rag_mvp.git
cd rag_mvp

# 2. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Create .env file and add your OpenAI key
touch .env
echo "OPENAI_API_KEY=sk-your-api-key-here" >> .env
# Replace sk-your-api-key-here with your actual key
# Get your key from: https://platform.openai.com/api-keys
```

> ⚠️ `.env` is not committed to the repo for security. You must create it yourself.

---

### Run Chunking Experiments

Each file in `chunk-lab/` is standalone. Run one by one to see how each technique works:

```bash
cd chunk-lab

# 1. Fixed size — cuts every N characters
python fixed_size.py

# 2. Fixed size + overlap — same but with overlapping edges
python recursive_char_text_split.py

# 3. Token-based — cuts every N tokens (how LLMs read)
python token_based.py

# 4. Similarity-based — cuts where meaning changes
python similarity_based.py

# 5. Header-based — cuts by markdown headers
python document-based/header_based.py

# 6. PDF-based — cuts by PDF structure (titles, headers)
python document-based/pdf_based.py
```

> 💡 Run them in order (1 → 6). Each one builds on the previous concept.
> Compare the output of each to see how chunks differ.

---

### Experience RAG Output

After running a chunking experiment that saves a vector DB (3, 4, 5, or 6), try asking questions:

```bash
cd app
python ask.py
```

```
Ask a question (type 'exit' to quit)

You: How many leaves can an employee take?
AI: Employees are entitled to 20 days of paid leave per year.

You: Can I work from home?
AI: Work from home is allowed up to 3 days a week with manager approval.

You: exit
```

> ⚠️ Experiments 1 and 2 only print chunks — they don't save a vector DB. Run experiment 3+ first before using `ask.py`.

> 💡 This is where chunking matters — better chunks = better answers.
> Try different chunking experiments, then run `ask.py` each time to see how answers change.

---

### Chunking Techniques Explained

---

#### 1. Fixed Size Chunking

File: `chunk-lab/fixed_size.py`

Slice text like cutting a rope every 500 characters — doesn't care if it cuts a word or sentence in half.

`"Hello world this is a test"` → `["Hello wor", "ld this i", "s a test"]` (if chunk_size=9)

```python
for i in range(0, len(text), chunk_size):
    chunk_text = text[i:i + chunk_size]
```

`chunk_size = 500` → every 500 chars, make a cut. That's it.

**Used when:** MVPs, simple text files, fast experimentation

**Limit:** No understanding of structure or meaning. Can split mid-sentence.

---

#### 2. Fixed Size + Overlapping Chunking

File: `chunk-lab/recursive_char_text_split.py`

Same as #1, but each chunk peeks into the next — so the end of chunk 1 repeats at the start of chunk 2.

`"ABCDEFGHIJ"` → `["ABCDEF", "EFGHIJ"]` (chunk_size=6, overlap=2 → "EF" appears in both)

Why? Without overlap you lose context at the edges. With overlap, no sentence falls through the crack.

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

`chunk_size = 500` → max 500 chars per chunk

`chunk_overlap = 100` → last 100 chars of chunk N = first 100 chars of chunk N+1

**Used when:** Q&A systems, policies, manuals

**Limit:** Increased storage & cost. Still no understanding of meaning.

---

#### 3. Structure-Aware Chunking

Files: `chunk-lab/document-based/`

Splits based on document structure, not character count.

**a) Header-Based (Markdown)**

File: `chunk-lab/document-based/header_based.py`

Splits markdown by headers — each section becomes a chunk.

```python
MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)
```

**b) PDF-Based**

File: `chunk-lab/document-based/pdf_based.py`

Parses PDF into elements (Title, Header, Text) and groups by section.

```python
UnstructuredPDFLoader("file.pdf", mode="elements")
```

`mode="elements"` → structure-aware parsing (titles, headers, paragraphs)

Needs: `brew install poppler`

**Used when:** Structured documents (PDFs, markdown, HTML), policy docs with clear sections

**Limit:** Depends on document having proper structure. PDF parsing can miss headers in scanned docs.

---

#### 4. Semantic (Similarity-Based) Chunking

File: `chunk-lab/similarity_based.py`

Splits by meaning — groups sentences that talk about the same topic.

LangChain's `SemanticChunker` does all the math internally (embed → compare → break).

How it works under the hood:

1. Split text into tiny sentence-level pieces
2. Embed each sentence using OpenAI
3. Compare consecutive sentences using cosine similarity
4. Where similarity drops = topic changed = chunk boundary

```python
SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=30
)
```

`breakpoint_threshold_type = "percentile"` → use percentile to decide what's a "low" similarity

`breakpoint_threshold_amount = 30` → bottom 30% similarity scores = topic change = chunk boundary

**Used when:** Text has no headers or structure, need topic-coherent chunks

**Limit:** Slow (needs to embed every sentence). More API calls = more cost. Needs: `pip install langchain-experimental`

---

#### 5. Token-Based Chunking

File: `chunk-lab/token_based.py`

Splits by token count, not character count — because LLMs read tokens, not characters.

**What is a token?**

A token is a **subword unit** defined by the tokenizer (tiktoken). It splits by spelling patterns, not by meaning or word boundaries.

Token ≠ Word. Rough rule: 1 token ≈ 4 characters ≈ ¾ of a word.

```
"Employees"     → 1 token    (simple word)
"unhappiness"   → 2 tokens   ["un", "happiness"]
"ChatGPT"       → 2 tokens   ["Chat", "GPT"]
"I'm"           → 2 tokens   ["I", "'m"]
```

**Why not just use character chunking?**

- `chunk_size = 500 chars` → could be 80 tokens or 150 tokens (unpredictable)
- `chunk_size = 50 tokens` → always exactly ≤ 50 tokens (guaranteed)
- LLMs have **token limits** not character limits: GPT-4 = 128K tokens, Embedding API = 8,191 tokens

```python
TokenTextSplitter(
    chunk_size=50,       # max 50 TOKENS per chunk (not characters!)
    chunk_overlap=10     # last 10 TOKENS repeat in next chunk
)
```

Character vs Token split:

```
Text: "Sick leave requires a medical certificate"

Character (chunk_size=30): "Sick leave requires a medical " | "certificate"
                                              ↑ cuts at 30 chars blindly

Token (chunk_size=6): "Sick leave requires a medical certificate"
                                              ↑ 6 tokens, clean boundary
```

**Used when:** Feeding chunks to LLMs or embedding APIs, need precise token limits

**Limit:** Slightly slower (runs tiktoken tokenizer). Needs: `pip install tiktoken`

---

### Summary

| #   | Method           | File                             | Splits by            |
| --- | ---------------- | -------------------------------- | -------------------- |
| 1   | Fixed Size       | `fixed_size.py`                  | Character count      |
| 2   | Fixed + Overlap  | `recursive_char_text_split.py`   | Characters + overlap |
| 3a  | Header-Based     | `document-based/header_based.py` | Markdown headers     |
| 3b  | PDF-Based        | `document-based/pdf_based.py`    | PDF structure        |
| 4   | Similarity-Based | `similarity_based.py`            | Semantic meaning     |
| 5   | Token-Based      | `token_based.py`                 | Token count          |

Simple → Smart: 1 → 2 → 5 → 3 → 4
