import os

from notetool.tool.path import exist_and_create


class WorkApp:
    def __init__(self, app_name='notedata', dir_app=None):
        self.dir_app = dir_app or f'/opt/notechats/apps/{app_name}'

        self.dir_db = os.path.join(self.dir_app, 'databases')
        self.dir_log = os.path.join(self.dir_app, 'logs')
        self.dir_common = os.path.join(self.dir_app, 'common')

    def create(self):
        exist_and_create(self.dir_app)
        exist_and_create(self.dir_db)
        exist_and_create(self.dir_log)
        exist_and_create(self.dir_common)

    def db_file(self, file_name='data.db'):
        return os.path.join(self.dir_db, file_name)

    def log_file(self, file_name='info.log'):
        return os.path.join(self.dir_log, file_name)

    def common_file(self, file_name='temp.txt'):
        return os.path.join(self.dir_common, file_name)


def db_file(app_name='notedata', file_name='data.db'):
    return WorkApp(app_name=app_name).db_file(file_name)


def log_file(app_name='notedata', file_name='data.db'):
    return WorkApp(app_name=app_name).log_file(file_name)
