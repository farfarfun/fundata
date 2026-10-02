# fundata

一批用于下载/管理机器学习常见公开数据集的脚本：数据集的名称、分类、下载地址等元信息存入本地 SQLite（`fundata.manage.core.DatasetManage`），实际数据文件从蓝奏网盘下载。收录的数据集包括 iris、MovieLens（100k/1m/10m/20m/25m）、Criteo（sample/kaggle）、Amazon Electronics 评论、adult、porto-seguro、bitly-usagov、COCO（val2017/annotations）以及 YOLOv3/v4 权重等。

`fundata.work`、`fundata.manage`、`fundata.tables_bak` 可以正常 import 和使用。`DatasetManage.download()` 中蓝奏云下载分支会抛出 `NotImplementedError`：当前 `fundrive` 的蓝奏云 API 是基于类的 `fundrive.drives.lanzou.LanZouDrive`，需要已认证的实例，没有免认证的下载函数可用。

`fundata.dataset` 可以在不安装重型机器学习依赖时导入；构建 Criteo 数据集时才需要安装 `dataset` extra。`tensorflow`、`keras`、`scikit-learn`、`funkeras`（提供 `notekeras`）已收录进 `pyproject.toml` 的 `dataset` extra，默认不随主依赖安装。

- `notekeras`：PyPI 上的 `funkeras` 包实际发布的顶层可 import 模块名仍然是 `notekeras`（`pip install funkeras` 装出来的目录是 `notekeras/`），因此代码里 `from notekeras.features.feature_parse import ...` 这一行即使装了 `funkeras` 也必须保持 `notekeras` 这个导入名不变，才能正确工作。

## 安装

已发布到 PyPI：

```bash
pip install fundata
```

需要用到 `fundata.dataset` 下具体数据集处理类时，额外装上 `dataset` extra：

```bash
pip install "fundata[dataset]"
```

基础包支持 Python 3.10；`dataset` extra 的机器学习依赖需要 Python 3.11 或更高版本，
以确保安装包含安全修复的 Keras 版本。

`pyproject.toml` 中已声明的 `dependencies` 为 `numpy`、`pandas`、`funshell`、`farlog`、`tqdm`，安装后 `fundata`（顶层）、`fundata.work`、`fundata.manage`、`fundata.tables_bak`、`fundata.dataset` 均可正常导入。构建 Criteo 数据集时需要额外安装 `dataset` extra。

## 用法示例

创建并查询本地数据集索引：

```python
from fundata.manage.core import DatasetManage
from fundata.manage.library import insert_library

dataset = DatasetManage(db_path="./fundata.db")
dataset.create()
insert_library(dataset)  # 把内置下载地址写入当前目录的 sqlite 索引
records = dataset.select(condition={"name": "movielens-100k"})
print(records[0]["urls"])
```

索引中的蓝奏云地址目前不能直接下载：`DatasetManage.download()` 遇到此类记录会抛出
`NotImplementedError`，需先接入已认证的 `fundrive.drives.lanzou.LanZouDrive` 实例。

管理数据落盘目录（数据库/日志/公共文件），默认落在 `/opt/farfarfun/apps/fundata`：

```python
from fundata.work import WorkApp

app = WorkApp()  # app_name 默认为 "fundata"，dir_app 默认为 /opt/farfarfun/apps/fundata
app.create()
```

`fundata.dataset` 下还有针对具体数据集的处理类，例如 `ElectronicsData`（Amazon 评论数据的下载、清洗、构建训练集）和 `CriteoData`（Criteo 数据集的特征处理）。

## 现状说明

`fundata.work` / `fundata.manage` / `fundata.tables_bak` / `fundata.dataset` 可正常导入；构建 Criteo 数据集时需要额外安装 `dataset` extra。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
