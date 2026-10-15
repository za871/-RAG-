from sqlalchemy import create_engine
from sqlalchemy.orm.session import sessionmaker

from backend.tests.sqlalchemy.Demo1 import User

DB_USER="root"
DB_PASSWORD="123456"
DB_HOST="127.0.0.1"
DB_PORT="3306"
DB_NAME="sqlalchemy_demo"
db_url=f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
engine=create_engine(db_url,echo=True)

SessionFactory=sessionmaker(bind=engine)
session=SessionFactory()
user=User(username="admin",password="123456")
session.add(user)
session.commit()
session.close()
