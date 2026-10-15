from backend.tests.sqlalchemy.utils.sqlalchemy_config import Base,engine
from backend.tests.sqlalchemy.utils.base_model import Account # noqa

if __name__ == '__main__':
    Base.metadata.create_all(engine)