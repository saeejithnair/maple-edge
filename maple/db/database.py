from sqlmodel import SQLModel, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession
from contextlib import asynccontextmanager

# Not directly used but this must be imported prior to calling
# SQLModel.metadata.create_all so that the models are registered.
from maple.db import models


USERNAME = "maple_admin"
PASSWORD = "vip_maple"
HOST = "localhost"
DATABASE = "maple_test_db"

def get_database_url(username: str = USERNAME, password: str = PASSWORD,
                     host: str = HOST, database: str = DATABASE) -> str:
    return f"postgresql+asyncpg://{username}:{password}@{host}/{database}"

async def create_db_and_tables(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

@asynccontextmanager
async def get_session(engine: AsyncEngine) -> AsyncSession:
    async with AsyncSession(engine) as session:
        yield session

def create_engine(database_url: str) -> AsyncEngine:
    return create_async_engine(
        database_url,
        future=True,
    )

async def init_db(database: str = DATABASE) -> AsyncEngine:
    database_url = get_database_url(database=database)
    async_engine = create_engine(database_url)
    await create_db_and_tables(engine=async_engine)

    return async_engine
