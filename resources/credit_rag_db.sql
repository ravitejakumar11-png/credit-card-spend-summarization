-- First create the db named “credit_rag_db” in pg admin 

-- =============================================================================
-- Enable pgvector — adds the VECTOR data type and <=> cosine distance operator
-- =============================================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- documents table
-- Every PDF you ingest gets one row here.
-- The UUID becomes a foreign key that links all chunks extracted from that PDF.
-- ON CONFLICT (filename) means re-ingesting the same file updates the row
-- instead of creating a duplicate — this is called an "upsert".
-- =============================================================================
CREATE TABLE IF NOT EXISTS documents (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT        UNIQUE NOT NULL,
    source_path TEXT        NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- =============================================================================
-- multimodal_chunks table
-- Every piece of content extracted from a document becomes one row here.
-- chunk_type tells us whether this was a text paragraph, a table, or an image.
--
-- image_path: Instead of storing raw image bytes in Postgres (wasteful), we
--   save images as PNG files in data/images/ and only keep the file path here.
--
-- embedding: A 1536-dimensional float vector. The <=> operator on this column
--   lets pgvector find the most similar chunks to a query vector.
--
-- metadata: A JSONB catch-all for any extra fields not worth a dedicated column.
-- =============================================================================
CREATE TABLE IF NOT EXISTS multimodal_chunks (
    id           BIGSERIAL    PRIMARY KEY,
    doc_id       UUID         NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    chunk_type   TEXT         NOT NULL CHECK (chunk_type IN ('text', 'table', 'image')),
    element_type TEXT,

    content      TEXT         NOT NULL,
    image_path   TEXT,
    mime_type    TEXT,

    page_number  INT,
    section      TEXT,
    source_file  TEXT,
    position     JSONB,

    embedding    VECTOR(1536),
    metadata     JSONB
);

-- =============================================================================
-- Indexes — these make searches fast
-- =============================================================================

-- HNSW (Hierarchical Navigable Small World) index for approximate nearest
-- neighbour search on the embedding column. "vector_cosine_ops" means we
-- use cosine similarity as the distance metric.
-- Without this index, every query does a full table scan — extremely slow.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON multimodal_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id    ON multimodal_chunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON multimodal_chunks (chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_page_number ON multimodal_chunks (page_number);






