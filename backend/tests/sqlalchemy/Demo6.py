from backend.tests.sqlalchemy.utils.base_model import User
from backend.tests.sqlalchemy.utils.sqlalchemy_config import SessionFactory

with SessionFactory() as session:
    #1
    user_asc=session.query(User).order_by(User.id).all()
    user_desc=session.query(User).order_by(User.id.desc()).all()
    #2
    page=2
    page_size=2
    users_page=session.query(User).order_by(User.id).limit(page_size).offset((page-1)*page_size).all()
    #3
    user_count=session.query(User).filter(User.username.like("%z%")).count()
