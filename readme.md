## RAG_MVP (Using LangChain)

Just a random project to understand RAG's chunking techniques

All chunking experiments live inside `chunk-lab/`

---

### 1. Fixed Size Chunking

    File: chunk-lab/fixed_size.py

    Slice text like cutting a rope every 500 characters — doesn't care if it cuts a word or sentence in half

    `"Hello world this is a test"` → `["Hello wor", "ld this i", "s a test"]` (if chunk_size=9)

    ```
    for i in range(0, len(text), chunk_size):
        chunk_text = text[i:i + chunk_size]
    ```

    `chunk_size = 500` → every 500 chars, make a cut. That's it.

    Used when:

        - MVPs
        - Simple text files
        - Fast experimentation

    Limit:

        - No understanding of structure or meaning
        - Can split mid-sentence

---

### 2. Fixed Size + Overlapping Chunking

    File: chunk-lab/recursive_char_text_split.py

    Same as #1, but each chunk peeks into the next — so the end of chunk 1 repeats at the start of chunk 2

    `"ABCDEFGHIJ"` → `["ABCDEF", "EFGHIJ"]` (chunk_size=6, overlap=2 → "EF" appears in both)

    Why? Without overlap you lose context at the edges. With overlap, no sentence falls through the crack.

    ```
    RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    ```

    `chunk_size = 500` → max 500 chars per chunk
    `chunk_overlap = 100` → last 100 chars of chunk N = first 100 chars of chunk N+1

    Used when:

        - Q&A systems
        - Policies, manuals

    Limit:

        - Increased storage & cost
        - Still no understanding of meaning

---

### 3. Structure-Aware Chunking

    Files: chunk-lab/document-based/

    Splits based on document structure, not character count

    **a) Header-Based (Markdown)**

        File: chunk-lab/document-based/header_based.py

        Splits markdown by headers — each section becomes a chunk

        ```
        MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )
        ```

    **b) PDF-Based**

        File: chunk-lab/document-based/pdf_based.py

        Parses PDF into elements (Title, Header, Text) and groups by section

        ```
        UnstructuredPDFLoader("file.pdf", mode="elements")
        ```

        `mode="elements"` → structure-aware parsing (titles, headers, paragraphs)

        Then groups text between headers into section chunks

        Needs: `brew install poppler`

        Alternate: `PyPDFLoader` (simpler, no structure awareness)

    Used when:

        - Structured documents (PDFs, markdown, HTML)
        - Policy docs with clear sections

    Limit:

        - Depends on document having proper structure
        - PDF parsing can miss headers in scanned docs

---

### 4. Semantic (Similarity-Based) Chunking

    File: chunk-lab/similarity_based.py

    Splits by meaning — groups sentences that talk about the same topic

    LangChain's `SemanticChunker` does all the math internally (embed → compare → break)

    How it works under the hood:

        1. Split text into tiny sentence-level pieces
        2. Embed each sentence using OpenAI
        3. Compare consecutive sentences using cosine similarity
        4. Where similarity drops = topic changed = chunk boundary

    ```
    SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=30
    )
    ```

    `breakpoint_threshold_type = "percentile"` → use percentile to decide what's a "low" similarity
    `breakpoint_threshold_amount = 30` → bottom 30% similarity scores = topic change = chunk boundary

    Used when:

        - Text has no headers or structure
        - Need topic-coherent chunks
        - Quality matters more than speed

    Limit:

        - Slow (needs to embed every sentence)
        - More API calls = more cost
        - Needs: `pip install langchain-experimental`

---

### Summary

    | # | Method              | File                          | Splits by        |
    |---|---------------------|-------------------------------|------------------|
    | 1 | Fixed Size          | fixed_size.py                 | Character count  |
    | 2 | Fixed + Overlap     | recursive_char_text_split.py  | Characters + overlap |
    | 3a| Header-Based        | document-based/header_based.py| Markdown headers |
    | 3b| PDF-Based           | document-based/pdf_based.py   | PDF structure    |
    | 4 | Similarity-Based    | similarity_based.py           | Semantic meaning |

    Simple → Smart: 1 → 2 → 3 → 4
