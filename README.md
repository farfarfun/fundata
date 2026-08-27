# fundata

一批用于下载/管理机器学习常见公开数据集的脚本：数据集的名称、分类、下载地址等元信息存入本地 SQLite（`fundata.manage.core.DatasetManage`），实际数据文件从蓝奏网盘下载。收录的数据集包括 iris、MovieLens（100k/1m/10m/20m/25m）、Criteo（sample/kaggle）、Amazon Electronics 评论、adult、porto-seguro、bitly-usagov、COCO（val2017/annotations）以及 YOLOv3/v4 权重等。

这是从旧包 `notedata` 改名而来、但改名并不彻底的遗留代码：`fundata/dataset/core.py`、`fundata/dataset/images.py` 仍然 `from notedata.manage import ...`（即依赖自己改名前的旧包名），`fundata/work/core.py` 里的默认路径也还是 `/opt/notechats/apps/notedata`（notechats 是改名前的旧组织名）。`pyproject.toml` 中 `dependencies` 为空，但代码实际用到了 `pandas`、`tensorflow`、`scikit-learn`、`tqdm`、`demjson`、`notedata`、`notedrive`、`notebuild` 等包，这些都需要手动安装。目前该仓库处于未维护状态，直接 `pip install` 之后大概率无法直接跑通。

## 安装

已发布到 PyPI（目前仅有 `1.0.1` 一个版本，`summary` 还是占位文本）：

```bash
pip install fundata
```

注意：由于 `dependencies` 未在 `pyproject.toml` 中声明，上面的安装不会带上实际需要的第三方库，用到对应功能时需要自行补装（如 `pandas`、`notedata`、`notedrive` 等）。

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

`fundata.dataset` 下还有针对具体数据集的处理类，例如 `ElectronicsData`（Amazon 评论数据的下载、清洗、构建训练集）和 `CriteoData`（Criteo 数据集的特征处理），但它们内部直接 `import notedata`，当前无法独立运行。

## 现状说明

代码内部残留大量指向旧包名 `notedata` / 旧组织名 `notechats` 的引用，属于改名后未收尾、也没有持续维护的历史代码，不建议作为正式依赖引入新项目。
