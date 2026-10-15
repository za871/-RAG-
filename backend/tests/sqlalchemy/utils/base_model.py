from datetime import datetime
from sqlalchemy import create_engine, Column, String,Integer,DateTime, text
from backend.tests.sqlalchemy.utils.sqlalchemy_config import Base


class User(Base):
    __tablename__="user"
    id=Column(Integer, primary_key=True,autoincrement=True,comment="用户ID")
    username=Column(String(50),nullable=False,unique=True,comment="用户名")
    password=Column(String(100),nullable=False,unique=True,comment="密码")
    create_time = Column(DateTime,default=datetime.now,comment="创建时间")

    def __repr__(self):
        return f"User(id={self.id},username={self.username},password={self.password},create_time={self.create_time})"
class Account(Base):
    __tablename__="account"

    id=Column(Integer,primary_key=True,autoincrement=True,comment="用户id")
    age=Column(String(50),nullable=False,comment="年龄")