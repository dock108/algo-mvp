import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_engine(url=None):
    url = url or os.getenv("ALGO_DB_URL", "sqlite:///data/algo.db")
    return create_engine(url, echo=False, future=True)


SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
