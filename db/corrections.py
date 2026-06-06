from db.connection import get_pool


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS correction_events (
    id          SERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT now(),
    document_id VARCHAR(64),
    element_id  VARCHAR(64),
    user_final  TEXT
);
"""


async def ensure_table() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(_CREATE_TABLE_SQL)


async def store_corrections(document_id: str, corrections: list[dict]) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.executemany(
            "INSERT INTO correction_events (document_id, element_id, user_final) "
            "VALUES ($1, $2, $3)",
            [(document_id, c["element_id"], c["user_final"]) for c in corrections],
        )


async def get_few_shot_examples(limit: int = 10) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT element_id, user_final FROM correction_events "
            "ORDER BY created_at DESC LIMIT $1",
            limit,
        )
    return [dict(r) for r in rows]
