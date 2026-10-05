"""fundata 领域相关异常类型。

统一在这里定义，避免业务代码里到处 ``raise Exception(...)``。
"""


class FunDataError(Exception):
    """fundata 所有领域异常的基类。"""


class TableConfigError(FunDataError):
    """表结构/字段配置错误，例如未设置 ``columns``。"""


class TableQueryError(FunDataError):
    """执行 SQL 语句失败。"""


class DatasetNotFoundError(FunDataError):
    """请求的数据集不在本地索引中。"""


class DatasetDownloadError(FunDataError):
    """数据集下载失败，或记录里没有可用的下载地址。"""


class DatasetBuildError(FunDataError):
    """数据集构建结果不符合预期，例如切分出的样本数与统计值对不上。"""
