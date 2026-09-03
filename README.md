# fundata

一批用于下载/管理机器学习常见公开数据集的脚本：数据集的名称、分类、下载地址等元信息存入本地 SQLite（`fundata.manage.core.DatasetManage`），实际数据文件从蓝奏网盘下载。收录的数据集包括 iris、MovieLens（100k/1m/10m/20m/25m）、Criteo（sample/kaggle）、Amazon Electronics 评论、adult、porto-seguro、bitly-usagov、COCO（val2017/annotations）以及 YOLOv3/v4 权重等。

`fundata.work`、`fundata.manage`、`fundata.tables_bak` 可以正常 import 和使用。`DatasetManage.download()` 中蓝奏云下载分支会抛出 `NotImplementedError`：当前 `fundrive` 的蓝奏云 API 是基于类的 `fundrive.drives.lanzou.LanZouDrive`，需要已认证的实例，没有免认证的下载函数可用。

`fundata.dataset`（`dataset/datas.py`）默认无法独立 import：它直接依赖 `demjson`、`tensorflow`、`scikit-learn`、`notekeras`。其中 `tensorflow`、`scikit-learn`、`funkeras`（提供 `notekeras`）已收录进 `pyproject.toml` 的 `dataset` extra，默认不随主依赖安装。需要特别说明两点：

- `notekeras`：PyPI 上的 `funkeras` 包实际发布的顶层可 import 模块名仍然是 `notekeras`（`pip install funkeras` 装出来的目录是 `notekeras/`），因此代码里 `from notekeras.features.feature_parse import ...` 这一行即使装了 `funkeras` 也必须保持 `notekeras` 这个导入名不变，才能正确工作。
- `demjson`：PyPI 上唯一的 `demjson==2.2.4` 是 Python 2 时代遗留的包，其 `setup.py` 使用了 `setuptools>=58` 已移除的 `use_2to3` 参数，在任何现代构建工具链下都会构建失败，因此**没有**声明进 `dataset` extra（声明了会导致 `uv lock`/`pip install fundata[dataset]` 直接失败）。这是上游包已废弃导致的已知限制，需要用到 `CriteoData._create_criteo_dataset` 里 `demjson.encode(...)` 那部分功能时，需自行想办法安装/替换 `demjson`。

## 安装

已发布到 PyPI：

```bash
pip install fundata
```

需要用到 `fundata.dataset` 下具体数据集处理类时，额外装上 `dataset` extra：

```bash
pip install "fundata[dataset]"
```

`pyproject.toml` 中已声明的 `dependencies` 为 `pandas`、`funshell`、`farlog`、`tqdm`，安装后 `fundata`（顶层）、`fundata.work`、`fundata.manage`、`fundata.tables_bak` 均可正常使用。`fundata.dataset` 需要额外的 `dataset` extra（`tensorflow`、`scikit-learn`、`funkeras`），`demjson` 因上游已废弃需自行处理（见上文说明）。

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

`fundata.work` / `fundata.manage` / `fundata.tables_bak` 可正常使用；`fundata.dataset` 需要额外安装 `dataset` extra 才能独立运行，不建议在装上这个 extra 之前作为正式依赖引入新项目。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
