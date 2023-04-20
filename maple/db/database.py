from sqlmodel import SQLModel, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

USERNAME = "maple_admin"
PASSWORD = "vip_maple"
HOST = "localhost"
DATABASE = "maple_db"
DATABASE_URL = f"postgresql://{USERNAME}:{PASSWORD}@{HOST}/{DATABASE}"


engine = create_async_engine(DATABASE_URL)

async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all(engine))

async def get_session():
    async with AsyncSession(engine) as session:
        yield session