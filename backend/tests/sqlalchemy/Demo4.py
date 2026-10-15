from backend.tests.sqlalchemy.utils.base_model import User
from backend.tests.sqlalchemy.utils.sqlalchemy_config import SessionFactory

with SessionFactory() as session:
    #1
    user_all=session.query(User).all()
    for user in user_all:
        print(user)
    #2
    user_by=session.get(User,1)
    print(user_by)
   #3
    user_first=session.query(User).first()
    print(user_first)
    #4
    users=session.query(User.username,User.password).all()
    for tup in users:
        print(tup)
        print(tup[0])
        print(tup[1])