import json
import os

from farlog import getLogger

from ..tables_bak.core import SqliteTable

logger = getLogger(__name__)


class DatasetManage(SqliteTable):
    """管理数据集索引和下载信息。"""

    def __init__(
        self, table_name: str = "datasets", db_path: str | None = None, *args, **kwargs
    ) -> None:
        """初始化数据集索引。"""
        if db_path is None:
            db_path = os.path.abspath(os.path.dirname(__file__)) + "/dataset.db"

        super(DatasetManage, self).__init__(
            db_path=db_path, table_name=table_name, *args, **kwargs
        )
        self.columns = ["name", "category", "describe", "urls", "md5", "path", "size"]

    def create(self) -> None:
        """创建数据集索引表。"""
        self.execute(
            """
                create table if not exists {} (
                name                varchar(200)  primary key 
               ,category            varchar(200)  DEFAULT ('')
               ,describe            varchar(5000) DEFAULT ('')
               ,urls                varchar(200)  DEFAULT ('')
               ,md5                 varchar(200)  DEFAULT ('')
               ,path                varchar(200)  DEFAULT ('')           
               ,size                integer       DEFAULT (0)
        )
        """.format(self.table_name)
        )

    def update(self, properties: dict, condition: dict | None = None) -> None:
        """更新数据集记录。"""
        condition = condition or {"name": properties["name"]}
        super(DatasetManage, self).update(properties, condition=condition)

    def encode(self, properties: dict) -> dict:
        """将 URL 列表编码为 JSON。"""
        if "urls" in properties.keys():
            properties["urls"] = json.dumps(properties["urls"])
        return properties

    def decode(self, properties: dict) -> dict:
        """将数据库中的 URL 字段解码。"""
        if "urls" in properties.keys():
            properties["urls"] = json.loads(json.loads(properties["urls"]))
        return properties

    def download(
        self,
        name: str,
        path: str | None = None,
        overwrite: bool = True,
        path_root: str = "./download/",
    ) -> bool:
        """下载指定数据集；蓝奏云需配置 fundrive 驱动凭据。"""
        res = self.select_pd(
            "select urls,path from table_name where name='{name}'".format(name=name)
        )

        if len(res) == 0:
            logger.warning(f"数据集不存在: name={name}")
            return False

        for line in res.to_dict(orient="records"):
            line = self.decode(line)
            path = path or line["path"]
            path = path_root + path

            if "lanzou" in line["urls"]:
                raise NotImplementedError(
                    "fundrive 的蓝奏云驱动需要认证实例，当前记录未提供认证配置"
                )

        return True


def download(name: str, path: str | None = None) -> None:
    """下载指定名称的数据集。"""
    DatasetManage().download(name, path=path)
