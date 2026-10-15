import sys
from pathlib import Path

# 直接 `python scripts/init_db.py` 时 sys.path[0] 是 scripts/ 而不是 backend/，
# import app 会 ModuleNotFoundError；把 backend/ 补进来，从哪个目录执行都一样
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import init_database
from app import models # noqa


def main():
    init_database()
    print("数据库初始化完成")
if __name__=="__main__":
    main()