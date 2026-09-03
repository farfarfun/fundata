"""fundata 领域相关异常类型。

统一在这里定义，避免业务代码里到处 ``raise Exception(...)``。
"""


class FunDataError(Exception):
    """fundata 所有领域异常的基类。"""


class TableConfigError(FunDataError):
    """表结构/字段配置错误，例如未设置 ``columns``。"""


class TableQueryError(FunDataError):
    """执行 SQL 语句失败。"""
