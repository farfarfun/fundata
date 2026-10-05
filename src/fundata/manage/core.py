import json
import os
import shutil
import urllib.request
from typing import Any

from farlog import getLogger

from ..exceptions import DatasetDownloadError
from ..tables_bak.core import SqliteTable

logger = getLogger(__name__)


def default_db_path() -> str:
    """返回默认的索引 sqlite 文件路径。

    按 ``FUNDATA_INDEX_DB`` 环境变量 → 包目录下的 ``dataset.db``（仅当该目录可写时）
    → ``~/.fundata/dataset.db`` 的顺序解析。``pip install`` 之后包目录通常位于
    site-packages，普通用户没有写权限，直接往那里建库会 ``OperationalError``。

    :return: sqlite 文件的绝对路径
    """
    explicit = os.environ.get("FUNDATA_INDEX_DB")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))

    package_dir = os.path.abspath(os.path.dirname(__file__))
    package_db = os.path.join(package_dir, "dataset.db")
    if os.path.exists(package_db) or os.access(package_dir, os.W_OK):
        return package_db

    return os.path.join(os.path.expanduser("~"), ".fundata", "dataset.db")


class DatasetManage(SqliteTable):
    """管理数据集索引和下载信息。"""

    def __init__(
        self, table_name: str = "datasets", db_path: str | None = None, **kwargs: Any
    ) -> None:
        """初始化数据集索引。

        :param table_name: 索引表名
        :param db_path: sqlite 文件路径，不传则按 :func:`default_db_path` 解析
        :param kwargs: 透传给 ``SqliteTable`` 的其余关键字参数
        """
        if db_path is None:
            db_path = default_db_path()

        super().__init__(db_path=db_path, table_name=table_name, **kwargs)
        self.columns = ["name", "category", "describe", "urls", "md5", "path", "size"]

    def create(self) -> None:
        """创建数据集索引表。"""
        self.execute(
            f"""
                create table if not exists {self.table_name} (
                name                varchar(200)  primary key
               ,category            varchar(200)  DEFAULT ('')
               ,describe            varchar(5000) DEFAULT ('')
               ,urls                varchar(200)  DEFAULT ('')
               ,md5                 varchar(200)  DEFAULT ('')
               ,path                varchar(200)  DEFAULT ('')
               ,size                integer       DEFAULT (0)
        )
        """
        )

    def update(self, properties: dict, condition: dict | None = None) -> None:
        """更新数据集记录。"""
        condition = condition or {"name": properties["name"]}
        super().update(properties, condition=condition)

    def encode(self, properties: dict) -> dict:
        """把 ``urls`` 字段编码成 JSON 字符串。

        返回新字典、不原地修改入参：``insert_library()`` 对同一条记录先 ``insert``
        再 ``update``，原地修改会让 ``urls`` 被编码两次。

        :param properties: 待写库的记录
        :return: 编码后的记录副本
        """
        if "urls" not in properties:
            return properties
        encoded = dict(properties)
        encoded["urls"] = json.dumps(encoded["urls"])
        return encoded

    def decode(self, properties: dict) -> dict:
        """把库里的 ``urls`` 字段解码成字典。

        兼容历史上被重复编码过一次的旧数据（旧版 ``encode`` 原地修改入参，
        ``insert`` + ``update`` 两次调用会把同一条记录编码两遍）。

        :param properties: 从库里读出的记录
        :return: 解码后的记录副本
        """
        if "urls" not in properties:
            return properties
        decoded = dict(properties)
        urls = json.loads(decoded["urls"])
        if isinstance(urls, str):
            urls = json.loads(urls)
        decoded["urls"] = urls
        return decoded

    @staticmethod
    def _fetch(url: str, target: str) -> None:
        """把 ``url`` 的内容流式写入 ``target``，先落临时文件再原子改名。

        :param url: 直链下载地址
        :param target: 本地目标文件路径
        :raises DatasetDownloadError: 请求或写入失败时抛出
        """
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        temp = target + ".part"
        try:
            with (
                urllib.request.urlopen(url) as response,
                open(temp, "wb") as file_handle,
            ):
                shutil.copyfileobj(response, file_handle)
        except OSError as e:
            if os.path.exists(temp):
                os.remove(temp)
            logger.error(f"下载失败: url={url} target={target} error={e}")
            raise DatasetDownloadError(f"下载失败: url={url}") from e
        os.replace(temp, target)

    def download(
        self,
        name: str,
        path: str | None = None,
        overwrite: bool = True,
        path_root: str = "./download/",
    ) -> str | None:
        """下载指定数据集并返回本地文件路径。

        :param name: 数据集名称，对应索引表的 ``name`` 字段
        :param path: 覆盖索引记录里的相对路径
        :param overwrite: 本地文件已存在时是否重新下载
        :param path_root: 本地数据根目录
        :return: 本地文件路径；数据集不在索引中时返回 ``None``
        :raises NotImplementedError: 记录只有蓝奏云地址时抛出，需接入已认证的
            ``fundrive.drives.lanzou.LanZouDrive`` 实例
        :raises DatasetDownloadError: 记录没有可用地址，或直链下载失败时抛出
        """
        # name 来自调用方，必须用参数绑定下发，不能拼进 SQL 字符串。
        res = self.select_pd(
            f"select urls,path from {self.table_name} where name = ?", params=(name,)
        )

        if len(res) == 0:
            logger.warning(f"数据集不存在: name={name}")
            return None

        line = self.decode(res.to_dict(orient="records")[0])
        urls = line["urls"]
        target = os.path.join(path_root, path or line["path"])

        if os.path.exists(target) and not overwrite:
            logger.info(f"文件已存在，跳过下载: name={name} path={target}")
            return target

        source = urls.get("source")
        if source:
            self._fetch(source, target)
            logger.info(f"下载完成: name={name} path={target}")
            return target

        if "lanzou" in urls:
            raise NotImplementedError(
                "fundrive 的蓝奏云驱动需要认证实例，当前记录未提供认证配置"
            )

        raise DatasetDownloadError(f"数据集 {name} 没有可用的下载地址: {sorted(urls)}")


def download(name: str, path: str | None = None) -> str | None:
    """下载指定名称的数据集。

    :param name: 数据集名称
    :param path: 覆盖索引记录里的相对路径
    :return: 本地文件路径；数据集不在索引中时返回 ``None``
    """
    return DatasetManage().download(name, path=path)
