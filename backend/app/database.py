from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.config import get_settings

class Base(DeclarativeBase):
    pass

settings=get_settings()
engine=create_engine(
    url=settings.database_url,
    echo=True,
    pool_pre_ping=True,#使用连接前检查连接是否可用
)
SessionLocal=sessionmaker(
    bind=engine,
    expire_on_commit=False,# 表示提交后对象的属性仍然可以读取
)

#执行这个函数就是拿到了一个关于session的生成器对象
def get_session():
    with SessionLocal() as session:
        yield session
#get_session()后续会用于FastAPI依赖注入，这样每次请求后都会获得一个数据库会话，请求结束后自动关闭

def init_database():
    Base.metadata.create_all(bind=engine)