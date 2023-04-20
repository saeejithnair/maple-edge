from sqlmodel import SQLModel, create_engine

DATABASE_URL = "postgresql://maple_admin:vip_maple@localhost/maple_db"

engine = create_engine(DATABASE_URL)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
