from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

db_url="mysql+pymysql://root:123456@127.0.0.1:3306/sqlalchemy_demo?charset=utf8mb4"
engine=create_engine(db_url,echo=True)
SessionFactory=sessionmaker(bind=engine)
session=SessionFactory()
Base=declarative_base()