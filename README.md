# fundata

一批用于下载/管理机器学习常见公开数据集的脚本：数据集的名称、分类、下载地址等元信息存入本地 SQLite（`fundata.manage.core.DatasetManage`），实际数据文件从蓝奏网盘下载。收录的数据集包括 iris、MovieLens（100k/1m/10m/20m/25m）、Criteo（sample/kaggle）、Amazon Electronics 评论、adult、porto-seguro、bitly-usagov、COCO（val2017/annotations）以及 YOLOv3/v4 权重等。

`fundata.work`、`fundata.manage`、`fundata.tables_bak` 可以正常 import 和使用。`DatasetManage.download()` 中蓝奏云下载分支会抛出 `NotImplementedError`：当前 `fundrive` 的蓝奏云 API 是基于类的 `fundrive.drives.lanzou.LanZouDrive`，需要已认证的实例，没有免认证的下载函数可用。

`fundata.dataset`（`dataset/datas.py`）目前无法独立 import：它直接依赖 `demjson`、`tensorflow`、`scikit-learn`、`notekeras`，均不是本仓库声明的依赖，环境中也未安装。需要特别说明的是 `notekeras`：PyPI 上的 `funkeras` 包实际发布的顶层可 import 模块名仍然是 `notekeras`（`pip install funkeras` 装出来的目录是 `notekeras/`），因此代码里 `from notekeras.features.feature_parse import ...` 这一行即使装了 `funkeras` 也必须保持 `notekeras` 这个导入名不变，才能正确工作。

## 安装

已发布到 PyPI：

```bash
pip install fundata
```

`pyproject.toml` 中已声明的 `dependencies` 为 `pandas`、`funshell`、`funutil`、`tqdm`，安装后 `fundata`（顶层）、`fundata.work`、`fundata.manage`、`fundata.tables_bak` 均可正常使用。`fundata.dataset` 需要额外自行安装 `demjson`、`tensorflow`、`scikit-learn`、`funkeras`（见上文关于导入名的说明）。

## 用法示例

维护一份数据集索引，并按需从蓝奏云下载：

```python
from fundata.manage.core import DatasetManage
from fundata.manage.library import insert_library

dataset = DatasetManage()
dataset.create()
insert_library()  # 把 iris/movielens/criteo/coco/yolo 权重等的下载地址写入本地 sqlite

dataset.download("movielens-100k", path_root="./download/")
```

管理数据落盘目录（数据库/日志/公共文件），默认落在 `/opt/farfarfun/apps/fundata`：

```python
from fundata.work import WorkApp

app = WorkApp()  # app_name 默认为 "fundata"，dir_app 默认为 /opt/farfarfun/apps/fundata
app.create()
```

`fundata.dataset` 下还有针对具体数据集的处理类，例如 `ElectronicsData`（Amazon 评论数据的下载、清洗、构建训练集）和 `CriteoData`（Criteo 数据集的特征处理），但如上文所述，它们依赖未声明的第三方包，当前无法独立运行。

## 现状说明

`fundata.work` / `fundata.manage` / `fundata.tables_bak` 可正常使用；`fundata.dataset` 因缺失重型 ML 依赖仍无法独立运行，不建议在这部分功能齐备之前作为正式依赖引入新项目。
