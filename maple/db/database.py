from databases import Database
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://maple_admin:vip_maple@localhost/maple_db"

Base = declarative_base()

def create_session(db_url):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False,
                                bind=engine)
    return SessionLocal()

def create_database(db_url):
    database = Database(db_url)
    return database

database, engine, SessionLocal = create_session(DATABASE_URL)
Base.metadata.create_all(engine)