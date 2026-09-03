# Changelog

本项目的版本记录按版本倒序排列，每个版本分「新增」「修复」「变更」「废弃」四类。

## [Unreleased]

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
