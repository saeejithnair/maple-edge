from sqlmodel import SQLModel, create_engine
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
import databases
import sqlalchemy
from contextlib import asynccontextmanager


USERNAME = "maple_admin"
PASSWORD = "vip_maple"
HOST = "localhost"
DATABASE = "maple_db"
DATABASE_URL = f"postgresql+asyncpg://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"

# database = databases.Database(DATABASE_URL)
# metadata = sqlalchemy.MetaData()
connect_args = {}
# connect_args = {"check_same_thread": False}
engine = create_async_engine(DATABASE_URL, future=True, connect_args=connect_args)
# engine = create_async_engine(DATABASE_URL, connect_args=connect_args)
# engine = sqlalchemy.create_engine(
#     DATABASE_URL, connect_args={"check_same_thread": False}
# )
# metadata.create_all(engine)

# engine = create_engine(DATABASE_URL, connect_args=connect_args)

async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    # SQLModel.metadata.create_all(engine)

# def create_db_and_tables():
#     SQLModel.metadata.create_all(engine)

@asynccontextmanager
async def get_session():
    async with AsyncSession(engine) as session:
        yield session