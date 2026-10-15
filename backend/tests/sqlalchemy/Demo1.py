from datetime import datetime
from sqlalchemy import create_engine, Column, String,Integer,DateTime, text
from sqlalchemy.orm import declarative_base

DB_USER="root"
DB_PASSWORD="123456"
DB_HOST="127.0.0.1"
DB_PORT="3306"
DB_NAME="sqlalchemy_demo"

# 先连接 MySQL 服务器(不带库名),目标库不存在则先创建
server_engine=create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/",echo=True)
with server_engine.connect() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} DEFAULT CHARACTER SET utf8mb4"))
server_engine.dispose()

db_url=f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
engine=create_engine(db_url,echo=True)

Base=declarative_base()

class User(Base):
    __tablename__="user"
    id=Column(Integer, primary_key=True,autoincrement=True,comment="用户ID")
    username=Column(String(50),nullable=False,unique=True,comment="用户名")
    password=Column(String(100),nullable=False,unique=True,comment="密码")
    create_time = Column(DateTime,default=datetime.now,comment="创建时间")

    def __repr__(self):
        return f"User(id={self.id},username={self.username},password={self.password},create_time={self.create_time})"

Base.metadata.create_all(engine)
print("数据库表创建完成")