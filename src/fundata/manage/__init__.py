"""数据集索引的管理入口。"""

from .core import DatasetManage, default_db_path
from .library import insert_library

__all__ = ["DatasetManage", "default_db_path", "insert_library"]
