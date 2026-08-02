import os
from functools import lru_cache
from pathlib import Path
import psycopg2
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

def get_connection_string():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "ai_platform")
    user = os.getenv("POSTGRES_USER", "platform_app")
    pwd  = os.getenv("POSTGRES_PASSWORD", "change_me_local")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"

@lru_cache(maxsize=1)
def get_engine():
    return create_engine(get_connection_string())

def get_raw_connection():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB", "ai_platform")
    user = os.getenv("POSTGRES_USER", "platform_app")
    pwd  = os.getenv("POSTGRES_PASSWORD", "change_me_local")
    return psycopg2.connect(
        host=host, port=port, dbname=db, user=user, password=pwd
    )

def run_sql_file(path):
    sql = Path(path).read_text()
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print(f"applied {path}")