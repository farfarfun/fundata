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

from pathlib import Path

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
        f"('demo', 'cat', 'demo.bin', '{urls}')"
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
        """最小替身：与真实 ``DatasetManage.download`` 一样返回本地路径。"""

        def __init__(self):
            self.names = []

        def download(self, name, **kwargs):
            self.names.append(name)
            return f"./download/{name}"

    default_manager = Manager()
    monkeypatch.setattr(core, "DatasetManage", lambda: default_manager)
    function = getattr(core, function_name)
    function()
    assert default_manager.names == expected_names

    explicit_manager = Manager()
    function(explicit_manager)
    assert explicit_manager.names == expected_names


def _index_with_local_source(tmp_path, name, relative_path, payload):
    """在临时索引里登记一条指向本地 file:// 直链的记录，返回管理器。

    用真实的 ``DatasetManage`` 和真实的 ``urlopen``（file:// 协议），不打桩下载，
    这样 ``download()`` 真的写文件、真的返回路径。
    """
    from fundata.manage.core import DatasetManage

    source = tmp_path / f"{name}.source"
    source.write_text(payload, encoding="utf-8")

    dataset = DatasetManage(db_path=str(tmp_path / "index.db"))
    dataset.create()
    dataset.insert(
        {
            "name": name,
            "category": "dataset",
            "urls": {"source": source.as_uri()},
            "path": relative_path,
        }
    )
    return dataset


def test_download_source_url_writes_file_and_returns_path(tmp_path):
    """直链记录会真的把内容写到本地，并返回本地路径。"""
    dataset = _index_with_local_source(tmp_path, "demo", "sub/demo.txt", "hello\n")
    try:
        target = dataset.download("demo", path_root=str(tmp_path / "download"))
        assert target == str(tmp_path / "download" / "sub" / "demo.txt")
        assert Path(target).read_text(encoding="utf-8") == "hello\n"
        # 临时文件不残留
        assert not Path(target + ".part").exists()
    finally:
        dataset.close()


def test_download_skips_existing_file_when_not_overwrite(tmp_path):
    """overwrite=False 且本地已存在时不重新拉取，内容保持不变。"""
    dataset = _index_with_local_source(tmp_path, "demo", "demo.txt", "new\n")
    try:
        root = tmp_path / "download"
        target = root / "demo.txt"
        target.parent.mkdir(parents=True)
        target.write_text("old\n", encoding="utf-8")

        assert dataset.download("demo", overwrite=False, path_root=str(root)) == str(
            target
        )
        assert target.read_text(encoding="utf-8") == "old\n"

        assert dataset.download("demo", overwrite=True, path_root=str(root)) == str(
            target
        )
        assert target.read_text(encoding="utf-8") == "new\n"
    finally:
        dataset.close()


def test_download_unknown_dataset_returns_none(tmp_path):
    """索引里没有的名字返回 None，而不是假装成功。"""
    dataset = _index_with_local_source(tmp_path, "demo", "demo.txt", "x")
    try:
        assert dataset.download("not-in-index", path_root=str(tmp_path)) is None
    finally:
        dataset.close()


def test_download_without_usable_url_raises(tmp_path):
    """既没有直链也没有蓝奏云地址时报错，不返回成功。"""
    from fundata.exceptions import DatasetDownloadError
    from fundata.manage.core import DatasetManage

    dataset = DatasetManage(db_path=str(tmp_path / "index.db"))
    dataset.create()
    dataset.insert({"name": "demo", "urls": {"other": "x"}, "path": "demo.txt"})
    try:
        with pytest.raises(DatasetDownloadError):
            dataset.download("demo", path_root=str(tmp_path))
    finally:
        dataset.close()


def test_download_broken_source_raises_and_cleans_temp(tmp_path):
    """直链不可达时抛领域异常，并且不留下半截文件。"""
    from fundata.exceptions import DatasetDownloadError
    from fundata.manage.core import DatasetManage

    dataset = DatasetManage(db_path=str(tmp_path / "index.db"))
    dataset.create()
    missing = tmp_path / "nope.bin"
    dataset.insert(
        {"name": "demo", "urls": {"source": missing.as_uri()}, "path": "demo.bin"}
    )
    root = tmp_path / "download"
    try:
        with pytest.raises(DatasetDownloadError):
            dataset.download("demo", path_root=str(root))
        assert not (root / "demo.bin").exists()
        assert not (root / "demo.bin.part").exists()
    finally:
        dataset.close()


def test_encode_does_not_mutate_input_and_decode_round_trips(tmp_path):
    """encode 不改入参，insert+update 之后 urls 仍然只编码了一层。"""
    from fundata.manage.core import DatasetManage

    dataset = DatasetManage(db_path=str(tmp_path / "index.db"))
    dataset.create()
    record = {"name": "demo", "urls": {"source": "http://a"}, "path": "demo.bin"}
    try:
        dataset.insert(record)
        dataset.update(record)
        assert record["urls"] == {"source": "http://a"}, "encode 不应原地修改入参"

        raw = dataset.select("select * from table_name")[0]
        assert dataset.decode(raw)["urls"] == {"source": "http://a"}
    finally:
        dataset.close()


def test_decode_tolerates_legacy_double_encoded_urls(tmp_path):
    """旧库里被编码两层的 urls 仍然能解码出字典。"""
    import json

    from fundata.manage.core import DatasetManage

    dataset = DatasetManage(db_path=str(tmp_path / "index.db"))
    try:
        legacy = json.dumps(json.dumps({"source": "http://a"}))
        assert dataset.decode({"urls": legacy})["urls"] == {"source": "http://a"}
    finally:
        dataset.close()


def test_get_adult_data_reads_both_downloaded_files(tmp_path):
    """Adult 入口真的读到下载下来的两个文件，并返回两个 DataFrame。"""
    from fundata.dataset import core
    from fundata.manage.core import DatasetManage

    train_src = tmp_path / "train.csv"
    train_src.write_text("1,a\n2,b\n", encoding="utf-8")
    test_src = tmp_path / "test.csv"
    # 还原 UCI adult.test 的形状：第一行是 `|...` 注释行（必须被 comment="|" 跳过），
    # 最后一行多一列（必须被 on_bad_lines="skip" 跳过）
    test_src.write_text("|1x3 Cross validator\n3,c\n4,d\n5,e,extra\n", encoding="utf-8")

    dataset = DatasetManage(db_path=str(tmp_path / "index.db"))
    dataset.create()
    dataset.insert(
        {
            "name": "adult-train",
            "urls": {"source": train_src.as_uri()},
            "path": "adult-data/adult.train.txt",
        }
    )
    dataset.insert(
        {
            "name": "adult-test",
            "urls": {"source": test_src.as_uri()},
            "path": "adult-data/adult.test.txt",
        }
    )
    try:
        train_data, test_data = core.get_adult_data(
            dataset, path_root=str(tmp_path / "download")
        )
        assert train_data.shape == (2, 2)
        assert test_data.shape == (2, 2)
        assert (tmp_path / "download" / "adult-data" / "adult.train.txt").exists()
    finally:
        dataset.close()


def test_get_adult_data_missing_index_raises(tmp_path):
    """索引里没有 adult 记录时抛 DatasetNotFoundError，而不是 AttributeError。"""
    from fundata.dataset import core
    from fundata.exceptions import DatasetNotFoundError
    from fundata.manage.core import DatasetManage

    dataset = DatasetManage(db_path=str(tmp_path / "index.db"))
    dataset.create()
    try:
        with pytest.raises(DatasetNotFoundError):
            core.get_adult_data(dataset, path_root=str(tmp_path / "download"))
    finally:
        dataset.close()


def test_to_csv_accepts_dict_condition(tmp_path):
    """字典条件导出 csv 时必须生成合法 SQL，只导出匹配的行。"""
    from fundata.tables_bak.core import SqliteTable

    table = SqliteTable(
        db_path=str(tmp_path / "csv.db"), table_name="demo", columns=["id", "name"]
    )
    try:
        table.execute("create table demo (id varchar(50), name varchar(50))")
        table.insert({"id": "1", "name": "alice"})
        table.insert({"id": "2", "name": "bob"})

        out = table.to_csv({"id": "1"}, path=str(tmp_path / "out.csv"))
        content = Path(out).read_text(encoding="utf-8")
        assert "alice" in content
        assert "bob" not in content
    finally:
        table.close()


def test_sqlite_table_accepts_bare_filename(tmp_path, monkeypatch):
    """db_path 不带目录时不能因为 makedirs('') 崩掉。"""
    from fundata.tables_bak.core import SqliteTable

    monkeypatch.chdir(tmp_path)
    table = SqliteTable(db_path="bare.db", table_name="demo", columns=["id"])
    try:
        assert (tmp_path / "bare.db").exists()
    finally:
        table.close()
