import json
import os

from notedrive.lanzou import download as download_lanzou
from notetool.database import SqliteTable


class DatasetManage(SqliteTable):
    def __init__(self, table_name='datasets', db_path=None, *args, **kwargs):
        if db_path is None:
            db_path = os.path.abspath(
                os.path.dirname(__file__)) + '/dataset.db'

        super(DatasetManage, self).__init__(
            db_path=db_path, table_name=table_name, *args, **kwargs)
        self.columns = ['name', 'category',
                        'describe', 'urls', 'md5', 'path', 'size']

    def create(self):
        self.execute("""
                create table if not exists {} (
                name                varchar(200)  primary key 
               ,category            varchar(200)  DEFAULT ('')
               ,describe            varchar(5000) DEFAULT ('')
               ,urls                varchar(200)  DEFAULT ('')
               ,md5                 varchar(200)  DEFAULT ('')
               ,path                varchar(200)  DEFAULT ('')           
               ,size                integer       DEFAULT (0)
        )
        """.format(self.table_name))

    def update(self, properties: dict, condition: dict = None):
        condition = condition or {'name': properties['name']}
        super(DatasetManage, self).update(properties, condition=condition)

    def encode(self, properties: dict):
        if 'urls' in properties.keys():
            properties['urls'] = json.dumps(properties['urls'])
            # properties['urls'] = demjson.encode(properties['urls'])
        return properties

    def decode(self, properties: dict):
        if 'urls' in properties.keys():
            properties['urls'] = json.loads(json.loads(properties['urls']))
        return properties

    def download(self, name, path=None, overwrite=True, path_root='./download/'):
        res = self.select_pd(
            "select urls,path from table_name where name='{name}'".format(name=name))

        if len(res) == 0:
            print("No this dataset")
            return False

        for line in res.to_dict(orient='records'):
            line = self.decode(line)
            path = path or line['path']
            path = path_root+path

            if 'lanzou' in line['urls'].keys():
                download_lanzou(line['urls']['lanzou'],
                                dir_pwd=os.path.dirname(path))

        return True


def download(name, path=None):
    DatasetManage().download(name, path=path)
