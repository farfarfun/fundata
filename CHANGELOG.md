# Changelog

本项目的版本记录按版本倒序排列，每个版本分「新增」「修复」「变更」「废弃」四类。

## [1.0.5]

### 新增

- `DatasetManage.download()` 实现 `source` 直链下载：流式写临时文件后原子改名，返回本地文件路径；只登记蓝奏云地址的记录仍抛 `NotImplementedError`，没有任何可用地址时抛 `DatasetDownloadError`。此前这条分支什么都不做就返回 `True`，调用方以为下载成功。
- `src/fundata/exceptions.py` 新增 `DatasetNotFoundError`/`DatasetDownloadError`。
- `pyproject.toml` 增加 `authors`、`[tool.ruff]`（`target-version = "py310"`）与 `[tool.pytest.ini_options]` 配置。
- 各子包 `__init__.py` 显式声明 `__all__`，明确重导出的公开 API。

### 修复

- `get_adult_data()` 原先把 `download()` 的返回值当成带 `.path` 属性的对象用，必然 `AttributeError`；且传了 pandas 2.0 起已删除的 `error_bad_lines` 参数，必然 `TypeError`。现在按返回的路径读文件、改用 `on_bad_lines="skip"`，并用 `comment="|"` 跳过 UCI `adult.test` 的首行注释（否则 pandas 会按它推断成 1 列、把全部正常行当坏行丢掉），返回 `(训练集, 测试集)` 两个 DataFrame；索引里没有记录时抛 `DatasetNotFoundError`。
- `BaseTable.to_csv()`/`pop_to_csv()` 传字典条件时会把 `dict` 直接拼进 SQL（`where {'id': '1'}`），生成的语句非法；新增 `_where_clause()` 统一处理字典/字符串条件，`delete()` 也改用它。
- `DatasetManage.encode()` 原地修改入参，导致 `insert_library()` 先 `insert` 再 `update` 把同一条记录的 `urls` 编码两层；改为返回副本，`decode()` 单层解码并兼容历史上被编码两层的旧数据。
- `library.check()` 按下标取 `data[0][3]`，而 `select_all()` 返回的是字典列表，必然 `KeyError`；改为走 `decode()`。
- `SqliteTable.__init__()` 在 `db_path` 不含目录时执行 `os.makedirs("")`，直接 `FileNotFoundError`；改为仅在有父目录时创建。
- `DatasetManage.__init__()` 的 `super().__init__(db_path=..., table_name=..., *args)` 属关键字参数后再解包位置参数，任何位置参数都会与 `db_path` 冲突报错；改为只透传关键字参数。
- `requires-python` 下限保持为组织统一的 `>=3.10`；`dataset` extra 中 TensorFlow/Keras 链路通过环境标记限定为 Python 3.11+（keras 3.13 起才要求 3.11，3.10 上只能解析到仍在安全公告受影响区间内的版本），`scikit-learn` 与 Keras 无关，不加标记。
- 显式声明源码直接导入的 `numpy` 运行时依赖。
- `description` 从 `Add your description here` 占位文案改为真实项目描述。

### 变更

- `_util.py`/`dataset/images.py`/`manage/library.py`/`dataset/datas.py` 补齐类型标注与中文 docstring。
- `dataset/images.py` 的 `json.load(open(...))` 改为 `with` 语句，去掉被下一行立刻覆盖的 `image_path` 死代码。
- `tests/test_smoke.py` 去掉把错误契约写死的替身测试（原来假设 `download()` 返回带 `.path` 的对象），改为用 `file://` 直链跑真实下载链路；新增下载成功/跳过已存在/未登记/无可用地址/下载失败清理临时文件、编解码幂等、`to_csv` 字典条件、裸文件名 `db_path` 等用例。
- `pyproject.toml` 与 `uv.lock` 同步重新生成。

### 废弃

- 无

## [1.0.4]

### 新增

- `WorkApp` 支持通过 `FUNDATA_APP_DIR` 环境变量覆盖默认数据目录。

### 修复

- 无

### 变更

- 无

### 废弃

- 无

## [1.0.3]

### 新增

- `pyproject.toml` 新增 `dataset` optional-dependencies extra，收录 `fundata.dataset` 实际用到但此前未声明的 `tensorflow`/`scikit-learn`/`funkeras`。`demjson` 因上游已废弃、无法在现代工具链下构建，未收录进 extra，已在 README/pyproject 注释中说明。
- 新增 `src/fundata/exceptions.py`，提供 `TableConfigError`/`TableQueryError` 领域异常。

### 修复

- `pandas`、`tqdm` 补上版本下限，避免解析到过旧版本。
- `SqliteTable.execute()` 不再吞掉 SQL 执行异常并 `print`，改为记录带表名/路径/SQL 的错误日志后抛出 `TableQueryError`。
- `BaseTable` 中裸 `raise Exception(...)` 改为 `NotImplementedError`（抽象方法）或 `TableConfigError`（字段未配置）。
- `tables_bak/core.py`、`manage/core.py`、`manage/library.py`、`dataset/core.py`、`dataset/datas.py` 中的 `print()` 诊断输出与 `funutil.getLogger` 统一改为 `farlog.getLogger`。

### 变更

- `tables_bak/core.py` 移除 `from typing import List`，公开方法改用 `list[str]`/`X | None` 等 3.10 内置泛型写法，并补齐缺失的参数、返回值类型标注。
- `pyproject.toml` 增加 `license = "MIT"` 声明。

### 废弃

- 无

## [1.0.2]

### 新增

- 初始可用版本：`fundata.manage`（SQLite 数据集索引）、`fundata.work`（数据落盘目录管理）、`fundata.tables_bak`（通用 SQLite 表封装）、`fundata.dataset`（具体数据集处理类）。

### 修复

- 无

### 变更

- 依赖从 `notedata`/`notetool`/`notedrive` 迁移到 `fun*`/本地等价实现。
- 补充 PEP 561 `py.typed` 标记。

### 废弃

- 无
