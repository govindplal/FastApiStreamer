import asyncio
from core.database import AsyncSessionLocal
from core.memory import store_memory, semantic_search

SAMPLE_OBSERVATIONS = [
    "The client requested all database passwords to follow the 16-character entropy rule.",
    "Python asyncio uses an event loop running in a single OS thread.",
    "Playwright failed to launch Chromium because system dependencies were missing.",
    "Postgres 16 default port is 5432 and uses pgvector for high-dimensional vectors.",
    "Tokyo office reported heavy monsoon rainfall with temperatures dropping to 18°C.",
    "User prefers dark mode interfaces with minimalist, paper-like typography.",
    "The Redis cache TTL was configured to expire session tokens after 3600 seconds.",
    "FastAPI routes using StreamingResponse yield Server-Sent Event payloads.",
    "The project repository is hosted on GitHub under the govindplal namespace.",
    "The top Hacker News post discussed open-source engines running on Apple Silicon."
]

async def run_test():
    async with AsyncSessionLocal() as session:
        print("1. Seeding 10 memory observations...")
        for obs in SAMPLE_OBSERVATIONS:
            await store_memory(obs, session)
        print("   Done seeding.")

        # Test query with zero matching keywords
        query = "What were the weather conditions in Japan?"
        print(f"\n2. Testing Semantic Search for: '{query}'")
        results = await semantic_search(query, session, top_k=2)
        
        for idx, entry in enumerate(results, 1):
            print(f"   [{idx}] {entry.content}")

if __name__ == "__main__":
    asyncio.run(run_test())