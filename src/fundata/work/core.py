import os

from .._util import exist_and_create


class WorkApp:
    def __init__(self, app_name="fundata", dir_app=None):
        # 默认路径 /opt/farfarfun/... 需要 root/系统级写权限；非特权环境可通过
        # FUNDATA_APP_DIR 环境变量整体覆盖，或直接传入 dir_app 参数。
        self.dir_app = dir_app or os.environ.get("FUNDATA_APP_DIR") or f"/opt/farfarfun/apps/{app_name}"

        self.dir_db = os.path.join(self.dir_app, "databases")
        self.dir_log = os.path.join(self.dir_app, "logs")
        self.dir_common = os.path.join(self.dir_app, "common")

    def create(self):
        exist_and_create(self.dir_app)
        exist_and_create(self.dir_db)
        exist_and_create(self.dir_log)
        exist_and_create(self.dir_common)

    def db_file(self, file_name="data.db"):
        return os.path.join(self.dir_db, file_name)

    def log_file(self, file_name="info.log"):
        return os.path.join(self.dir_log, file_name)

    def common_file(self, file_name="temp.txt"):
        return os.path.join(self.dir_common, file_name)


def db_file(app_name="fundata", file_name="data.db"):
    return WorkApp(app_name=app_name).db_file(file_name)


def log_file(app_name="fundata", file_name="data.db"):
    return WorkApp(app_name=app_name).log_file(file_name)
