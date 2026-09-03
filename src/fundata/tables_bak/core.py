import os
import sqlite3
import time
from time import strftime
from typing import Any

import pandas as pd
from farlog import getLogger
from funshell import run_shell

from ..exceptions import TableConfigError, TableQueryError


class BaseTable:
    """
    表维度的底层数据库的通用实现
    """

    def __init__(
        self, table_name: str = "default_table", columns: list[str] | None = None
    ) -> None:
        """
        初始化一个通用数据库
        :param table_name: 表名
        :param columns: 字段列表
        """
        self.table_name = table_name
        self.columns = columns
        self.logger = getLogger(table_name)

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        """
        执行sql的引擎，需子类实现
        :param sql: sql
        :return: 执行的返回结果
        """
        raise NotImplementedError(
            f"{type(self).__name__}.execute() 未实现，需在子类（如 SqliteTable）中重写"
        )

    def insert(self, properties: dict) -> Any:
        """
        插入单条记录，当表设置唯一键插入时，如果唯一键已存在，则返回
        :param properties: 记录以字典形式保存，key是字段名，value是字段值
        :return: 插入 成功or失败
        """
        properties = self.encode(properties)
        keys, values = self._properties2kv(properties)

        sql = """insert or ignore into {table_name} ({columns}) values ({value})""".format(
            table_name=self.table_name, columns=", ".join(keys), value=", ".join(values)
        )
        return self.execute(sql)

    def update(self, properties: dict, condition: dict) -> Any:
        """
        更新数据
        :param properties: 需要更新的字段
        :param condition:  where条件
        :return: 更新 成功or失败
        """
        properties = self.encode(properties)
        equal = self._condition2equal(properties)
        equal2 = self._condition2equal(condition)
        sql = """update  {} set {} where {}""".format(
            self.table_name, ", ".join(equal), " and ".join(equal2)
        )
        return self.execute(sql)

    def update_or_insert(self, properties: dict, condition: dict | None = None) -> Any:
        """
        更新或者插入，首先尝试更新，更新失败则插入
        :param properties: 需要更新的字段
        :param condition:  where条件
        :return: 更新/插入 成功or失败
        """
        up = self.update(properties, condition)
        if up.rowcount == 0:
            return self.insert(properties)
        else:
            return up

    def decode(self, properties: dict) -> dict:
        """
        需要子类实现
        有些数据插入时可能需要编码/加密等特殊操作，同时，取数据后需要有对应的解码/解密操作，默认不编码/加密
        :param properties: 记录数据
        :return: 编码/加密后的数据
        """
        return properties

    def encode(self, properties: dict) -> dict:
        """
        需要子类实现
        有些数据插入时可能需要编码/加密等特殊操作，同时，取数据后需要有对应的解码/解密操作，默认不解码/解密
        :param properties: 编码/加密后记录数据
        :return: 解码/加密后的数据
        """
        return properties

    def count(self, properties: dict | None) -> int:
        """
        满足条件的数据量
        :param properties: 记录数
        :return: 记录条数
        """
        properties = properties or {}
        values = []
        for key in self.columns:
            value = str(properties.get(key, ""))
            if len(value) > 0 and len(key) > 0:
                values.append("{}='{}'".format(key, value))

        sql = """select count(1) from {} where {}""".format(
            self.table_name, " and ".join(values)
        )

        rows = self.execute(sql)
        for row in rows:
            return row[0]
        return 0

    def select_all(self) -> list[dict]:
        """
        返回全表数据
        """
        return self.select("select * from table_name")

    def select(self, sql: str | None = None, condition: dict | None = None) -> list[dict]:
        """
        根据sql或者指定条件选择数据
        :param sql: sql
        :param condition: 条件
        :return: 记录list
        """
        if sql is None:
            equal2 = self._condition2equal(condition)
            sql = """select * from {} where {}""".format(
                self.table_name, " and ".join(equal2)
            )
        else:
            sql = self.sql_format(sql)

        rows = self.execute(sql)
        return [] if rows is None else [dict(zip(self.columns, row)) for row in rows]

    def _properties2kv(self, properties: dict) -> tuple[list[str], list[str]]:
        """
        将输入的记录数据转换成表的字段名和字段值数据
        :param properties: 记录数据
        :return: keys and values
        """
        if self.columns is None:
            raise TableConfigError(
                f"表 {self.table_name} 未设置 columns，无法转换记录字段"
            )
        keys = []
        values = []
        for key in self.columns:
            value = str(properties.get(key, "")).replace("'", "")
            if len(key) > 0 and len(value) > 0:
                keys.append(key)
                values.append("'{}'".format(value))
        return keys, values

    def _condition2equal(self, properties: dict | str) -> list[str] | str:
        """
        将输入的记录数据转换成表的字段名和字段值数据的等式
        :param properties: 记录数据
        :return: 等式
        """
        if isinstance(properties, str):
            return properties
        if self.columns is None:
            raise TableConfigError(
                f"表 {self.table_name} 未设置 columns，无法构造 where 条件"
            )
        equals = []
        for key in self.columns:
            value = properties.get(key, None)
            if len(key) > 0 and value is not None:
                if isinstance(value, str):
                    equals.append("{}='{}'".format(key, value))
                else:
                    equals.append("{}={}".format(key, value))
        return equals

    def sql_format(self, sql: str) -> str:
        """
        对sql进行格式化，如一些特定字符串的替换
        :param sql: sql
        :return: 格式化之后的sql
        """
        sql = sql.replace("table_name", self.table_name)
        return sql

    def delete(self, condition: dict | str | None = None) -> None:
        """
        删除表
        :param condition:
        :return:
        """
        if condition is None:
            sql = "delete from {}".format(self.table_name)
        elif isinstance(condition, str):
            sql = "delete from {} where {}".format(self.table_name, condition)
        elif isinstance(condition, dict):
            sql = "delete from {} where {}".format(
                self.table_name, self._condition2equal(condition)
            )
        else:
            sql = None
        if sql is not None:
            self.execute(sql)
            self.logger.info("delete records with sql = {}".format(sql))


class SqliteTable(BaseTable):
    def __init__(
        self, db_path: str, conn: sqlite3.Connection | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """
        基于 sqlite3 的表实现。
        :param db_path: 数据库文件路径
        :param conn: 已建立的连接，不传则按 db_path 新建
        """
        super(SqliteTable, self).__init__(*args, **kwargs)
        self.db_path = db_path
        if not os.path.exists(os.path.dirname(self.db_path)):
            os.makedirs(os.path.dirname(self.db_path))
        self.conn = conn or sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.logger.info("db path:{}".format(self.db_path))

    def execute(self, sql: str, commit: bool = True, *args: Any, **kwargs: Any) -> sqlite3.Cursor:
        """
        sql执行核心
        :param commit: 是否需要commit
        :param sql: 执行sql
        :return: 执行结果
        :raises TableQueryError: sql 执行失败时抛出，带上下文信息，不再静默吞掉
        """
        try:
            rows = self.cursor.execute(sql)
            if commit:
                self.conn.commit()
            return rows
        except sqlite3.Error as e:
            self.logger.error(
                "执行 SQL 失败: table={} db_path={} sql={} error={}".format(
                    self.table_name, self.db_path, sql, e
                )
            )
            raise TableQueryError(
                f"表 {self.table_name} 执行 SQL 失败: {e}"
            ) from e

    def close(self) -> None:
        """
        关闭数据库连接
        """
        self.cursor.close()
        self.conn.close()

    def select_pd(self, sql: str = "select * from table_name") -> pd.DataFrame:
        """
        将表数据转成pandas的DataFrame
        :param sql: sql
        :return: DataFrame
        """
        sql = self.sql_format(sql)
        return pd.read_sql(sql, self.conn)

    def save_and_truncate(self) -> pd.DataFrame:
        """
        将全表数据导出为 csv 后清空表，并执行 VACUUM 回收空间。
        :return: 导出前的全表数据
        """
        result = pd.read_sql("select * from {}".format(self.table_name), self.conn)

        count = len(result)
        path = "{}/{}-{}-{}".format(
            os.path.dirname(self.db_path),
            self.table_name,
            count,
            strftime("%Y%m%d#%H:%M:%S", time.localtime()),
        )
        result.to_csv(path)
        self.logger.info("save to csv:{}->{}".format(count, path))

        self.execute("delete from {}".format(self.table_name))
        self.logger.info("delete from {}".format(self.table_name))
        self.vacuum()
        return result

    def to_csv(
        self,
        condition: str | dict | None,
        path: str | None = None,
        pop: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        将满足条件的数据导出为 csv。
        :param condition: where 条件
        :param path: 导出文件路径，不传则按表名和时间戳生成
        :param pop: 导出后是否删除并 VACUUM
        :return: 导出文件路径
        """
        if condition is None:
            sql = "select * from {}".format(self.table_name)
        else:
            sql = "select * from {} where {}".format(self.table_name, condition)

        path = path or (
            "{}/{}-{}".format(
                os.path.dirname(self.db_path),
                self.table_name,
                strftime("%Y%m%d#%H:%M:%S", time.localtime()),
            )
        )
        self.logger.info("save to csv -> {}".format(path))

        cmd = 'sqlite3 -header -csv {db_path} "{sql};" > {path}'.format(
            db_path=self.db_path, path=path, sql=sql
        )
        run_shell(cmd)
        if pop:
            self.delete(condition)
            self.vacuum()
        return path

    def pop_to_csv(self, condition: str | dict | None, path: str | None = None) -> str:
        """
        导出满足条件的数据为 csv，并从表中删除这些数据。
        :param condition: where 条件
        :param path: 导出文件路径
        :return: 导出文件路径
        """
        return self.to_csv(condition, path=path, pop=True)

    def vacuum(self) -> None:
        """
        数据库清理
        """
        self.execute("VACUUM")
        self.logger.info("数据库VACUUM")

    def insert_list(self, property_list: list[dict]) -> bool:
        """
        批量插入
        :param property_list: 待插入的记录列表
        :return: 是否成功
        """
        values = [
            tuple([properties.get(key, "") for key in self.columns])
            for properties in property_list
        ]
        sql = "insert or ignore into {} values ({})".format(
            self.table_name, ",".join(["?"] * len(self.columns))
        )

        self.cursor.executemany(sql, values)
        self.conn.commit()
        return True
