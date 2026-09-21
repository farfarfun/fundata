import os

from .._util import exist_and_create


class WorkApp:
    """管理 fundata 的应用目录。"""

    def __init__(self, app_name: str = "fundata", dir_app: str | None = None) -> None:
        """初始化应用目录配置。"""
        # 默认路径 /opt/farfarfun/... 需要 root/系统级写权限；非特权环境可通过
        # FUNDATA_APP_DIR 环境变量整体覆盖，或直接传入 dir_app 参数。
        self.dir_app = dir_app or os.environ.get("FUNDATA_APP_DIR") or f"/opt/farfarfun/apps/{app_name}"

        self.dir_db = os.path.join(self.dir_app, "databases")
        self.dir_log = os.path.join(self.dir_app, "logs")
        self.dir_common = os.path.join(self.dir_app, "common")

    def create(self) -> None:
        """创建数据库、日志和公共文件目录。"""
        exist_and_create(self.dir_app)
        exist_and_create(self.dir_db)
        exist_and_create(self.dir_log)
        exist_and_create(self.dir_common)

    def db_file(self, file_name: str = "data.db") -> str:
        """返回数据库文件路径。"""
        return os.path.join(self.dir_db, file_name)

    def log_file(self, file_name: str = "info.log") -> str:
        """返回日志文件路径。"""
        return os.path.join(self.dir_log, file_name)

    def common_file(self, file_name: str = "temp.txt") -> str:
        """返回公共文件路径。"""
        return os.path.join(self.dir_common, file_name)


def db_file(app_name: str = "fundata", file_name: str = "data.db") -> str:
    """返回应用数据库文件路径。"""
    return WorkApp(app_name=app_name).db_file(file_name)


def log_file(app_name: str = "fundata", file_name: str = "data.db") -> str:
    """返回应用日志文件路径。"""
    return WorkApp(app_name=app_name).log_file(file_name)
