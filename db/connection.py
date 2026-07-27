import asyncpg
from app.config import settings

_pool = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(settings.database_url)
    return _pool


async def close_pool() -> None:
    """Release Postgres connections on shutdown. No-op if the pool was never
    created (e.g. the DB was never reached, or this process only ever served
    /process). Called from app.main's lifespan shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
