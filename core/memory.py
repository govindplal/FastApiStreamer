from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.db import MemoryEntry

# Load the local Nomic model (768 dimensions)
# trust_remote_code=True is required for Nomic architectures
embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

def embed(text: str) -> list[float]:
    """Converts a string of text into a 768-dimensional semantic vector."""
    # Prefixing with 'search_document:' is a Nomic-specific optimization 
    prefixed_text = f"search_document: {text}"
    vector = embedder.encode(prefixed_text)
    return vector.tolist()

async def semantic_search(query: str, db: AsyncSession, top_k: int = 5):
    """Finds the most semantically similar memories using Cosine Distance."""
    # Prefixing queries helps the model map questions to the document space
    query_vector = embed(f"search_query: {query}")
    
    # pgvector's cosine distance operator is `<=>`
    # We order by distance ascending (closest meaning first)
    stmt = (
        select(MemoryEntry)
        .order_by(MemoryEntry.embedding.cosine_distance(query_vector))
        .limit(top_k)
    )
    
    result = await db.execute(stmt)
    return result.scalars().all()

async def store_memory(content: str, db: AsyncSession):
    """Embeds and saves an observation to long-term memory."""
    vector = embed(content)
    
    new_memory = MemoryEntry(
        content=content,
        embedding=vector
    )
    
    db.add(new_memory)
    await db.commit()