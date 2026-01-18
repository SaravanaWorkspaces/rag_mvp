## RAG_MVP

Just a random project to understand RAG's chunking technics

1.  Fixed size

    Chunks are sized using number of characters

    So chunks are approximately 500 characters long.

    Used when:

        - MVPs

        - Simple text files

        - Fast experimentation

    Limit:

        - No understanding of structure or meaning

2.  Fixed size + Overlapping Chunking

    Fixed-size chunks with overlapping content

    Example:

        ``chunk_size = 500
          chunk_overlap = 50``

    Used when:

        - Q&A systems

        - Policies, manuals

    Limit:

        - Increased storage & cost

3.  Structure-Aware Chunking
