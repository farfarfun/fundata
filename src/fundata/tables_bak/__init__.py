"""通用 sqlite 表封装。

TODO: consider consolidating with fundrive's table helpers, if/when it
has an equivalent.
"""

from .core import BaseTable, SqliteTable

__all__ = ["BaseTable", "SqliteTable"]
