from backend.tests.sqlalchemy.utils.base_model import User
from backend.tests.sqlalchemy.utils.sqlalchemy_config import SessionFactory
from sqlalchemy import and_,or_

with SessionFactory() as session:
    #1,
    user_1=session.query(User).filter(User.username=="admin").all()
    print(f"{user_1=}")
    #2
    user_2 = session.query(User).filter(User.username != "admin").all()
    print(f"{user_2=}")
    #3
    user_3 = session.query(User).filter(User.username.like("%ad%")).all()
    print(f"{user_3=}")
    #4
    user_4 = session.query(User).filter(User.id.in_([1,2,3])).all()
    print(f"{user_4=}")
    #5
    user_5 = session.query(User).filter(User.id > 2).all()
    print(f"{user_5=}")
    #6
    user_6=session.query(User).filter(and_(User.id>1,User.username.like("%w%"))).first()
    print(f"{user_6=}")
    #7
    user_7=session.query(User).filter(or_(User.id>1,User.username.like("%w%"))).first()
    print(f"{user_7=}")

