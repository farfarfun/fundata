"""表维度的轻量数据库封装。

本模块对外暴露 :class:`BaseTable` 与 :class:`SqliteTable`。

SQL 构造约定（重要）
--------------------
所有**字段值**一律走 sqlite 的参数绑定（``?`` 占位符 + 参数序列）下发，不再拼进 SQL
字符串；**标识符**（表名、字段名）无法参数化，因此在 :class:`BaseTable` 构造时就用
白名单（``[A-Za-z_][A-Za-z0-9_]*``）校验，非法标识符直接拒绝。
"""

import csv
import os
import re
import sqlite3
import time
from time import strftime
from typing import Any

import pandas as pd
from farlog import getLogger

from ..exceptions import TableConfigError, TableQueryError

# 表名/字段名的白名单：sqlite 的标识符无法用 ? 绑定，只能靠白名单挡住注入。
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# sql_format() 里 ``table_name`` 占位符的替换模式，加词边界避免误伤
# ``my_table_name`` 这类包含该子串的标识符。
_TABLE_NAME_PLACEHOLDER = re.compile(r"\btable_name\b")


def check_identifier(name: str, kind: str = "标识符") -> str:
    """校验一个 SQL 标识符（表名或字段名）是否安全可直接拼入 SQL。

    标识符不能用参数绑定下发，所以只接受 ``[A-Za-z_][A-Za-z0-9_]*``。

    :param name: 待校验的标识符
    :param kind: 出错信息里的称呼，例如 "表名" / "字段名"
    :return: 原样返回 ``name``，便于链式使用
    :raises TableConfigError: 标识符为空或含有白名单之外的字符
    """
    if not isinstance(name, str) or not _IDENTIFIER_RE.match(name):
        raise TableConfigError(
            f"非法的{kind}: {name!r}，只允许字母、数字和下划线，且不能以数字开头"
        )
    return name


class BaseTable:
    """
    表维度的底层数据库的通用实现
    """

    def __init__(
        self, table_name: str = "default_table", columns: list[str] | None = None
    ) -> None:
        """
        初始化一个通用数据库
        :param table_name: 表名，必须是合法标识符
        :param columns: 字段列表，每一项都必须是合法标识符
        :raises TableConfigError: 表名或字段名不合法
        """
        self.table_name = check_identifier(table_name, "表名")
        if columns is not None:
            for column in columns:
                check_identifier(column, "字段名")
        self.columns = columns
        self.logger = getLogger(table_name)

    def execute(
        self, sql: str, params: list | tuple | None = None, *args: Any, **kwargs: Any
    ) -> Any:
        """
        执行sql的引擎，需子类实现
        :param sql: 带 ``?`` 占位符的 sql
        :param params: 与占位符一一对应的参数序列
        :return: 执行的返回结果
        """
        raise NotImplementedError(
            f"{type(self).__name__}.execute() 未实现，需在子类（如 SqliteTable）中重写"
        )

    def require_columns(self) -> list[str]:
        """返回本表的字段列表，未配置时抛出领域异常。

        :return: 字段名列表
        :raises TableConfigError: 构造时没有传 ``columns``
        """
        if self.columns is None:
            raise TableConfigError(f"表 {self.table_name} 未设置 columns，无法构造 SQL")
        return self.columns

    def insert(self, properties: dict) -> Any:
        """
        插入单条记录，当表设置唯一键插入时，如果唯一键已存在，则返回
        :param properties: 记录以字典形式保存，key是字段名，value是字段值
        :return: 插入 成功or失败
        """
        properties = self.encode(properties)
        keys, values = self._properties2kv(properties)
        if not keys:
            raise TableConfigError(
                f"表 {self.table_name} 的 insert() 没有拿到任何可写字段，"
                f"传入的 key 与 columns={self.columns} 不匹配"
            )

        sql = "insert or ignore into {table_name} ({columns}) values ({value})".format(
            table_name=self.table_name,
            columns=", ".join(keys),
            value=", ".join(["?"] * len(keys)),
        )
        return self.execute(sql, values)

    def update(self, properties: dict, condition: dict | str) -> Any:
        """
        更新数据
        :param properties: 需要更新的字段
        :param condition:  where条件，字典或原始 where 字符串
        :return: 更新 成功or失败
        """
        properties = self.encode(properties)
        set_clause, set_params = self._condition2equal(properties)
        if not set_clause:
            raise TableConfigError(
                f"表 {self.table_name} 的 update() 没有拿到任何可更新字段，"
                f"传入的 key 与 columns={self.columns} 不匹配"
            )
        where_clause, where_params = self._require_where_clause(condition, "update")
        sql = "update {} set {} where {}".format(
            self.table_name, ", ".join(set_clause), where_clause
        )
        return self.execute(sql, [*set_params, *where_params])

    def update_or_insert(
        self, properties: dict, condition: dict | str | None = None
    ) -> Any:
        """
        更新或者插入，首先尝试更新，更新失败则插入
        :param properties: 需要更新的字段
        :param condition:  where条件，必传；缺省时无法判断该更新哪些行
        :return: 更新/插入 成功or失败
        :raises TableConfigError: 未提供 where 条件
        """
        if condition is None:
            raise TableConfigError(
                f"表 {self.table_name} 的 update_or_insert() 必须显式提供 condition，否则无法确定更新范围"
            )
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

    def count(self, properties: dict | None = None) -> int:
        """
        满足条件的数据量
        :param properties: 条件字典，不传或为空字典时统计全表
        :return: 记录条数
        """
        clauses, params = self._condition2equal(properties or {})
        if clauses:
            sql = f"select count(1) from {self.table_name} where " + " and ".join(
                clauses
            )
        else:
            sql = f"select count(1) from {self.table_name}"

        rows = self.execute(sql, params)
        for row in rows:
            return row[0]
        return 0

    def select_all(self) -> list[dict]:
        """
        返回全表数据
        """
        return self.select(f"select * from {self.table_name}")

    def select(
        self,
        sql: str | None = None,
        condition: dict | str | None = None,
        params: list | tuple | None = None,
    ) -> list[dict]:
        """
        根据sql或者指定条件选择数据
        :param sql: 原始 sql，可带 ``?`` 占位符（与 ``params`` 配套使用）
        :param condition: 条件字典或原始 where 字符串，仅在 ``sql`` 为空时生效
        :param params: ``sql`` 里占位符对应的参数序列
        :return: 记录list
        """
        if sql is None:
            where_clause, where_params = self._where_clause(condition or {})
            if where_clause:
                sql = f"select * from {self.table_name} where {where_clause}"
            else:
                sql = f"select * from {self.table_name}"
            params = where_params
        else:
            sql = self.sql_format(sql)

        rows = self.execute(sql, params)
        if rows is None:
            return []
        columns = self.require_columns()
        return [dict(zip(columns, row)) for row in rows]

    def _properties2kv(self, properties: dict) -> tuple[list[str], list[Any]]:
        """
        将输入的记录数据拆成字段名列表和与之对应的绑定参数列表
        :param properties: 记录数据
        :return: (字段名列表, 参数列表)，字段值不再拼进 SQL，由调用方配 ``?`` 占位符
        """
        keys = []
        values = []
        for key in self.require_columns():
            value = properties.get(key, "")
            if len(key) > 0 and value is not None and str(value) != "":
                keys.append(key)
                values.append(value)
        return keys, values

    def _condition2equal(
        self, properties: dict | str
    ) -> tuple[list[str] | str, list[Any]]:
        """
        将输入的记录数据转换成带 ``?`` 占位符的等式和对应的参数列表
        :param properties: 记录数据，或已经写好的原始 where 字符串
        :return: (等式列表或原始字符串, 参数列表)。传入字符串时原样返回且参数为空，
            由调用方自行保证该字符串不含外部输入。
        """
        if isinstance(properties, str):
            return properties, []
        equals = []
        params = []
        for key in self.require_columns():
            value = properties.get(key, None)
            if len(key) > 0 and value is not None:
                equals.append(f"{key}=?")
                params.append(value)
        return equals, params

    def _where_clause(self, condition: dict | str) -> tuple[str, list[Any]]:
        """把字典或字符串条件统一转换成 where 子句和绑定参数。

        :param condition: 字典条件（字段=值，多个字段用 and 连接）或原始 where 字符串
        :return: (where 子句字符串, 参数列表)
        """
        if isinstance(condition, str):
            return condition, []
        clauses, params = self._condition2equal(condition)
        return " and ".join(clauses), params

    def _require_where_clause(
        self, condition: dict | str, action: str
    ) -> tuple[str, list[Any]]:
        """构造 where 子句，并拒绝「条件为空」这种会误伤全表的调用。

        :param condition: 字典条件或原始 where 字符串
        :param action: 出错信息里的动作名，例如 "update" / "delete"
        :return: (where 子句字符串, 参数列表)
        :raises TableConfigError: 条件为空（空字典、空字符串，或 key 与 columns 完全不匹配）
        """
        clause, params = self._where_clause(condition)
        if not clause.strip():
            raise TableConfigError(
                f"表 {self.table_name} 的 {action}() 收到空 where 条件 {condition!r}，"
                f"为避免误伤全表已拒绝执行（delete/to_csv 要作用于全表请显式传 condition=None）"
            )
        return clause, params

    def sql_format(self, sql: str) -> str:
        """
        对sql进行格式化：把独立出现的 ``table_name`` 占位符替换成真实表名。
        只替换完整单词，``my_table_name`` 这类标识符不受影响。
        :param sql: sql
        :return: 格式化之后的sql
        """
        return _TABLE_NAME_PLACEHOLDER.sub(self.table_name, sql)

    def delete(self, condition: dict | str | None = None) -> None:
        """
        删除满足条件的记录，不传条件则清空表
        :param condition: 字典条件或原始 where 字符串
        :return: 无
        """
        if condition is None:
            sql, params = f"delete from {self.table_name}", []
        else:
            where_clause, params = self._require_where_clause(condition, "delete")
            sql = f"delete from {self.table_name} where {where_clause}"
        self.execute(sql, params)
        self.logger.info(f"delete records with sql = {sql}")


class SqliteTable(BaseTable):
    """基于 sqlite3 的表实现。"""

    def __init__(
        self,
        db_path: str,
        conn: sqlite3.Connection | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        基于 sqlite3 的表实现。
        :param db_path: 数据库文件路径
        :param conn: 已建立的连接，不传则按 db_path 新建
        """
        super().__init__(*args, **kwargs)
        self.db_path = db_path
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.conn = conn or sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.logger.info(f"db path:{self.db_path}")

    def execute(
        self,
        sql: str,
        params: list | tuple | None = None,
        commit: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> sqlite3.Cursor:
        """
        sql执行核心
        :param sql: 执行sql，字段值一律用 ``?`` 占位符
        :param params: 与占位符一一对应的参数序列
        :param commit: 是否需要commit
        :return: 执行结果
        :raises TableQueryError: sql 执行失败时抛出，带上下文信息，不再静默吞掉
        """
        try:
            rows = self.cursor.execute(sql, tuple(params or ()))
            if commit:
                self.conn.commit()
            return rows
        except sqlite3.Error as e:
            self.logger.error(
                f"执行 SQL 失败: table={self.table_name} db_path={self.db_path} sql={sql} error={e}"
            )
            raise TableQueryError(f"表 {self.table_name} 执行 SQL 失败: {e}") from e

    def close(self) -> None:
        """
        关闭数据库连接
        """
        self.cursor.close()
        self.conn.close()

    def select_pd(
        self, sql: str = "select * from table_name", params: list | tuple | None = None
    ) -> pd.DataFrame:
        """
        将表数据转成pandas的DataFrame
        :param sql: 原始 sql，可带 ``?`` 占位符
        :param params: 与占位符一一对应的参数序列
        :return: DataFrame
        """
        sql = self.sql_format(sql)
        return pd.read_sql(sql, self.conn, params=tuple(params or ()))

    def _default_export_path(self) -> str:
        """按表名和时间戳生成默认的导出文件路径。"""
        return "{}/{}-{}.csv".format(
            os.path.dirname(self.db_path) or ".",
            self.table_name,
            strftime("%Y%m%d-%H%M%S", time.localtime()),
        )

    def save_and_truncate(self) -> pd.DataFrame:
        """
        将全表数据导出为 csv 后清空表，并执行 VACUUM 回收空间。
        :return: 导出前的全表数据
        """
        result = pd.read_sql(f"select * from {self.table_name}", self.conn)

        count = len(result)
        path = self._default_export_path()
        result.to_csv(path, index=False)
        self.logger.info(f"save to csv:{count}->{path}")

        self.execute(f"delete from {self.table_name}")
        self.logger.info(f"delete from {self.table_name}")
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

        直接走 sqlite3 + pandas 读写，不再拼 shell 命令调用外部 ``sqlite3``
        可执行文件——原实现把 ``db_path``、where 条件和输出路径都拼进命令字符串，
        任何一个带空格或 shell 元字符都会被解释执行。

        :param condition: where 条件，字典或原始 where 字符串；``None`` 表示导出全表
        :param path: 导出文件路径，不传则按表名和时间戳生成
        :param pop: 导出后是否删除并 VACUUM
        :return: 导出文件路径
        """
        if condition is None:
            sql, params = f"select * from {self.table_name}", []
        else:
            where_clause, params = self._require_where_clause(condition, "to_csv")
            sql = f"select * from {self.table_name} where {where_clause}"

        path = path or self._default_export_path()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.logger.info(f"save to csv -> {path}")

        frame = pd.read_sql(sql, self.conn, params=tuple(params))
        # 与原先 `sqlite3 -header -csv` 的输出保持一致：带表头、不带行号索引。
        frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
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
        columns = self.require_columns()
        values = [
            tuple([properties.get(key, "") for key in columns])
            for properties in property_list
        ]
        sql = "insert or ignore into {} ({}) values ({})".format(
            self.table_name, ", ".join(columns), ",".join(["?"] * len(columns))
        )

        self.cursor.executemany(sql, values)
        self.conn.commit()
        return True
