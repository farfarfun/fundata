"""Lightweight smoke tests for the ``fundata`` package.

Background / import-path notes
-------------------------------
``fundata.work``, ``fundata.manage`` and ``fundata.tables_bak`` import
cleanly with only the declared dependencies (``pandas``, ``funshell``,
``farlog``, ``tqdm``): they use a local ``fundata._util`` helper for
small file-path utilities, and (for ``manage.core.DatasetManage.download``'s
lanzou branch, which has no drop-in replacement in the current
``fundrive`` API) raise ``NotImplementedError`` instead of failing at
import time.

``fundata.dataset`` imports without the optional ML stack; tests below cover
its lightweight public classes. Criteo processing remains guarded by the
``dataset`` extra and is tested separately in environments that install it.
"""

import logging

import pytest


def test_import_top_level_package():
    """`import fundata` must succeed with zero optional dependencies."""
    import fundata

    assert fundata.__name__ == "fundata"


def test_import_paths_submodule():
    """`fundata.paths` is a real, currently-empty submodule; must import cleanly."""
    import fundata.paths  # noqa: F401


def test_work_app_smoke():
    """fundata.work.WorkApp: construct + path helpers, real filesystem
    creation via fundata._util.exist_and_create."""
    import fundata.work as work

    app = work.WorkApp(app_name="smoke-test-app", dir_app="/tmp/fundata-smoke-app")
    assert app.dir_db == "/tmp/fundata-smoke-app/databases"
    assert app.dir_log == "/tmp/fundata-smoke-app/logs"
    assert app.db_file("data.db") == "/tmp/fundata-smoke-app/databases/data.db"
    assert app.log_file("info.log") == "/tmp/fundata-smoke-app/logs/info.log"
    assert app.common_file("temp.txt") == "/tmp/fundata-smoke-app/common/temp.txt"

    app.create()
    for d in (app.dir_app, app.dir_db, app.dir_log, app.dir_common):
        assert __import__("os").path.isdir(d)

    # module-level convenience functions
    assert work.db_file(app_name="smoke-test-app", file_name="d.db").endswith(
        "databases/d.db"
    )
    assert work.log_file(app_name="smoke-test-app", file_name="l.log").endswith(
        "logs/l.log"
    )


def test_tables_bak_base_table_smoke():
    """fundata.tables_bak.BaseTable: pure SQL-string-building logic."""
    from fundata.tables_bak.core import BaseTable

    table = BaseTable(table_name="demo", columns=["id", "name"])
    assert table.table_name == "demo"
    assert table.logger is not None

    keys, values = table._properties2kv({"id": "1", "name": "a"})
    assert keys == ["id", "name"]
    assert values == ["'1'", "'a'"]

    equal = table._condition2equal({"id": "1"})
    assert equal == ["id='1'"]

    assert table.sql_format("select * from table_name") == "select * from demo"

    # BaseTable.execute() is an abstract hook that must be implemented by
    # subclasses (e.g. SqliteTable); calling it directly is documented to
    # raise.
    with pytest.raises(Exception):
        table.execute("select 1")


def test_tables_bak_sqlite_table_crud_smoke(tmp_path):
    """fundata.tables_bak.SqliteTable against a throwaway local sqlite
    file under pytest's tmp_path (no network, no credentials, no shared
    state) -- exercises real insert/select/update logic.

    Includes dictionary-condition deletion against multiple rows.
    """
    from fundata.tables_bak.core import SqliteTable

    db_path = tmp_path / "smoke.db"
    table = SqliteTable(db_path=str(db_path), table_name="demo", columns=["id", "name"])
    try:
        table.execute(
            "create table if not exists demo (id varchar(50) primary key, name varchar(50))"
        )
        table.insert({"id": "1", "name": "alice"})
        rows = table.select("select * from demo")
        assert rows == [{"id": "1", "name": "alice"}]

        table.update({"name": "bob"}, condition={"id": "1"})
        rows = table.select("select * from demo")
        assert rows == [{"id": "1", "name": "bob"}]
    finally:
        table.close()


def test_tables_bak_delete_dict_condition(tmp_path):
    """字典条件能删除匹配记录，不影响不匹配记录。"""
    from fundata.tables_bak.core import SqliteTable

    table = SqliteTable(
        db_path=str(tmp_path / "delete.db"),
        table_name="demo",
        columns=["id", "name"],
    )
    try:
        table.execute("create table demo (id varchar(50), name varchar(50))")
        table.insert({"id": "1", "name": "alice"})
        table.insert({"id": "2", "name": "alice"})

        table.delete({"id": "1", "name": "alice"})
        assert table.select_all() == [{"id": "2", "name": "alice"}]

        table.delete({"id": "missing"})
        assert table.select_all() == [{"id": "2", "name": "alice"}]
    finally:
        table.close()


def test_manage_dataset_manage_smoke(tmp_path):
    """fundata.manage.DatasetManage imports cleanly (subclasses the local
    fundata.tables_bak.core.SqliteTable). Exercise real CRUD against a
    throwaway sqlite db; the lanzou-download branch still can't be smoke
    tested (needs real credentials/network) so it's left untouched here."""
    from fundata.manage.core import DatasetManage

    db_path = tmp_path / "datasets.db"
    dataset = DatasetManage(db_path=str(db_path))
    dataset.create()

    dataset.execute(
        "insert into datasets (name, category, urls) values ('demo', 'cat', '{}')"
    )
    rows = dataset.select("select * from datasets")
    assert len(rows) == 1
    assert rows[0]["name"] == "demo"
    dataset.close()


def test_insert_library_uses_explicit_manager(tmp_path):
    """内置索引可写入调用方指定的数据库。"""
    from fundata.manage import DatasetManage, insert_library

    dataset = DatasetManage(db_path=str(tmp_path / "library.db"))
    try:
        dataset.create()
        insert_library(dataset)
        rows = dataset.select(condition={"name": "movielens-100k"})
        assert len(rows) == 1
    finally:
        dataset.close()


def test_manage_lanzou_download_not_implemented(tmp_path):
    """未配置认证时，蓝奏云下载明确报告不可用。"""
    import json

    from fundata.manage.core import DatasetManage

    db_path = tmp_path / "datasets.db"
    dataset = DatasetManage(db_path=str(db_path))
    dataset.create()
    urls = json.dumps(json.dumps({"lanzou": "http://example.com/x"}))
    dataset.execute(
        "insert into datasets (name, category, path, urls) values "
        "('demo', 'cat', 'demo.bin', '{}')".format(urls)
    )

    with pytest.raises(NotImplementedError):
        dataset.download("demo")
    dataset.close()


def test_dataset_public_classes_import_and_base_contract(tmp_path):
    """数据集公开类可导入，基类契约在无网络环境下可执行。"""
    from fundata.dataset import CriteoData, ElectronicsData

    electronics = ElectronicsData(data_path=str(tmp_path))
    criteo = CriteoData(data_path=str(tmp_path))
    assert electronics.path_root == str(tmp_path)
    assert criteo.criteo_sample.endswith("criteo/criteo_sample.txt")
    with pytest.raises(NotImplementedError):
        electronics.build_dataset()


@pytest.mark.parametrize(
    ("function_name", "expected_names"),
    [
        (
            "get_movielens",
            [
                "movielens-100k",
                "movielens-1m",
                "movielens-10m",
                "movielens-20m",
                "movielens-25m",
            ],
        ),
        ("get_porto_seguro_data", ["porto-seguro-train", "porto-seguro-test"]),
        ("get_bitly_usagov_data", ["bitly-usagov"]),
    ],
)
def test_dataset_helpers_accept_default_and_explicit_manager(
    monkeypatch, function_name, expected_names
):
    """公开下载入口支持默认管理器和调用方显式传入的管理器。"""
    from fundata.dataset import core

    class Manager:
        def __init__(self):
            self.names = []

        def download(self, name, **kwargs):
            self.names.append(name)

    default_manager = Manager()
    monkeypatch.setattr(core, "DatasetManage", lambda: default_manager)
    function = getattr(core, function_name)
    function()
    assert default_manager.names == expected_names

    explicit_manager = Manager()
    function(explicit_manager)
    assert explicit_manager.names == expected_names


def test_get_adult_data_accepts_default_and_explicit_manager(monkeypatch):
    """Adult 入口支持两种管理器传入方式并读取两个下载结果。"""
    from types import SimpleNamespace

    from fundata.dataset import core

    class Manager:
        def __init__(self):
            self.names = []

        def download(self, name, **kwargs):
            self.names.append(name)
            return SimpleNamespace(path=f"/{name}.csv")

    read_paths = []
    monkeypatch.setattr(core.pd, "read_table", lambda path, **kwargs: read_paths.append(path))
    default_manager = Manager()
    monkeypatch.setattr(core, "DatasetManage", lambda: default_manager)

    core.get_adult_data()
    explicit_manager = Manager()
    core.get_adult_data(explicit_manager)

    assert default_manager.names == ["adult-train", "adult-test"]
    assert explicit_manager.names == ["adult-train", "adult-test"]
    assert read_paths == [
        "/adult-train.csv",
        "/adult-test.csv",
        "/adult-train.csv",
        "/adult-test.csv",
    ]
