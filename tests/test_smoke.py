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


def test_default_db_path_falls_back_when_package_dir_unwritable(tmp_path, monkeypatch):
    """包目录不可写时（pip 装进 site-packages 的常态）回落到用户目录，而不是直接报错。"""
    from fundata.manage import core as manage_core

    monkeypatch.delenv("FUNDATA_INDEX_DB", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    real_access = manage_core.os.access
    package_dir = str(Path(manage_core.__file__).parent)

    def fake_access(path, mode):
        if str(path) == package_dir:
            return False
        return real_access(path, mode)

    monkeypatch.setattr(manage_core.os, "access", fake_access)
    monkeypatch.setattr(manage_core.os.path, "exists", lambda p: False)

    assert manage_core.default_db_path() == str(
        tmp_path / "home" / ".fundata" / "dataset.db"
    )


def test_default_db_path_honours_env_override(tmp_path, monkeypatch):
    """FUNDATA_INDEX_DB 显式指定时优先生效。"""
    from fundata.manage import default_db_path

    monkeypatch.setenv("FUNDATA_INDEX_DB", str(tmp_path / "custom" / "idx.db"))
    assert default_db_path() == str(tmp_path / "custom" / "idx.db")


def test_module_level_log_file_default_matches_class_api():
    """模块级 log_file() 的默认文件名必须和 WorkApp.log_file() 一致，都是 info.log。"""
    import inspect

    import fundata.work as work

    assert (
        inspect.signature(work.log_file).parameters["file_name"].default
        == inspect.signature(work.WorkApp.log_file).parameters["file_name"].default
        == "info.log"
    )
    assert work.log_file(app_name="smoke-test-app").endswith("logs/info.log")
    assert work.db_file(app_name="smoke-test-app").endswith("databases/data.db")


def test_build_dataset_3_raises_instead_of_asserting(tmp_path):
    """切分结果对不上时抛领域异常，而不是用 assert（python -O 下会被整条删掉）。"""
    import inspect
    import pickle

    import pandas as pd

    from fundata.dataset.datas import ElectronicsData
    from fundata.exceptions import DatasetBuildError

    source = inspect.getsource(ElectronicsData.build_dataset_3)
    # 去掉注释行再判断，正文里解释「为什么不能用 assert」的那句注释不算数
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "assert " not in code

    data = ElectronicsData(data_path=str(tmp_path))
    reviews = pd.DataFrame(
        {"reviewerID": [0, 0, 0], "asin": [1, 2, 3], "unixReviewTime": [1, 2, 3]}
    )
    Path(data.pkl_remap).parent.mkdir(parents=True, exist_ok=True)
    with open(data.pkl_remap, "wb") as handle:
        pickle.dump(reviews, handle)
        pickle.dump([0, 0, 0, 0], handle)
        # user_count 故意写成 99，与实际只有 1 个用户对不上
        pickle.dump((99, 4, 1, 3), handle)

    with pytest.raises(DatasetBuildError):
        data.build_dataset_3()


def test_public_dataset_methods_have_chinese_docstrings():
    """ElectronicsData 的公开处理步骤和 CriteoDataBak 类都必须有中文 docstring。"""
    from fundata.dataset.datas import CriteoDataBak, ElectronicsData

    targets = [
        ElectronicsData.convert_pd_1,
        ElectronicsData.remap_id_2,
        ElectronicsData.build_dataset_3,
        CriteoDataBak,
    ]
    for target in targets:
        doc = (target.__doc__ or "").strip()
        assert doc, f"{target.__qualname__} 缺少 docstring"
        assert any("一" <= ch <= "鿿" for ch in doc), (
            f"{target.__qualname__} 的 docstring 不是中文"
        )


def test_preprocess_steps_respect_overwrite(tmp_path):
    """overwrite=True 必须真的重算，而不是因为目标文件已存在就直接跳过。"""
    import pickle

    import pandas as pd

    from fundata.dataset.datas import ElectronicsData

    data = ElectronicsData(data_path=str(tmp_path))
    Path(data.pkl_remap).parent.mkdir(parents=True, exist_ok=True)

    # 先放一份"旧结果"占位
    Path(data.pkl_dataset).write_bytes(b"stale")

    reviews = pd.DataFrame(
        {"reviewerID": [0, 0, 0], "asin": [1, 2, 3], "unixReviewTime": [1, 2, 3]}
    )
    with open(data.pkl_remap, "wb") as handle:
        pickle.dump(reviews, handle)
        pickle.dump([0, 0, 0, 0], handle)
        pickle.dump((1, 4, 1, 3), handle)

    # overwrite=False：保持旧内容不动
    data.build_dataset_3(overwrite=False)
    assert Path(data.pkl_dataset).read_bytes() == b"stale"

    # overwrite=True：重新生成
    data.build_dataset_3(overwrite=True)
    assert Path(data.pkl_dataset).read_bytes() != b"stale"


def test_tables_bak_base_table_smoke():
    """fundata.tables_bak.BaseTable: pure SQL-string-building logic."""
    from fundata.tables_bak.core import BaseTable

    table = BaseTable(table_name="demo", columns=["id", "name"])
    assert table.table_name == "demo"
    assert table.logger is not None

    # 字段值不再拼进 SQL，而是作为绑定参数原样返回
    keys, values = table._properties2kv({"id": "1", "name": "a"})
    assert keys == ["id", "name"]
    assert values == ["1", "a"]

    equal, params = table._condition2equal({"id": "1"})
    assert equal == ["id=?"]
    assert params == ["1"]

    assert table.sql_format("select * from table_name") == "select * from demo"
    # 只替换完整单词，含该子串的标识符不受影响
    assert table.sql_format("select my_table_name from table_name") == (
        "select my_table_name from demo"
    )

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


def test_to_csv_does_not_invoke_shell(tmp_path):
    """导出 csv 不再拼 shell 命令，改走 sqlite3 + pandas。"""
    import inspect

    from fundata.tables_bak import core as tables_core
    from fundata.tables_bak.core import SqliteTable

    source = inspect.getsource(tables_core)
    assert "run_shell" not in source
    assert "subprocess" not in source

    table = SqliteTable(
        db_path=str(tmp_path / "noshell.db"), table_name="demo", columns=["id", "name"]
    )
    try:
        table.execute("create table demo (id varchar(50), name varchar(50))")
        table.insert({"id": "1", "name": "alice"})
        out = table.to_csv(None, path=str(tmp_path / "all.csv"))
        lines = Path(out).read_text(encoding="utf-8").strip().splitlines()
        # 与原先 `sqlite3 -header -csv` 的输出一致：带表头、不带行号索引
        assert lines[0] == "id,name"
        assert lines[1] == "1,alice"
    finally:
        table.close()


def test_to_csv_path_with_shell_metacharacters(tmp_path):
    """导出路径带空格和 shell 元字符时必须按字面量写文件，而不是被 shell 解释。"""
    from fundata.tables_bak.core import SqliteTable

    table = SqliteTable(
        db_path=str(tmp_path / "meta.db"), table_name="demo", columns=["id", "name"]
    )
    try:
        table.execute("create table demo (id varchar(50), name varchar(50))")
        table.insert({"id": "1", "name": "alice"})

        weird = tmp_path / "out dir; touch pwned" / "a b.csv"
        out = table.to_csv(None, path=str(weird))
        assert Path(out) == weird
        assert weird.exists()
        assert not (tmp_path / "pwned").exists()
    finally:
        table.close()


def test_sql_injection_in_condition_value_is_not_executed(tmp_path):
    """where 条件值里的单引号只能当普通字符，不能提前闭合字符串改写 SQL。"""
    from fundata.tables_bak.core import SqliteTable

    table = SqliteTable(
        db_path=str(tmp_path / "inject.db"), table_name="demo", columns=["id", "name"]
    )
    try:
        table.execute("create table demo (id varchar(50), name varchar(50))")
        table.insert({"id": "1", "name": "alice"})
        table.insert({"id": "2", "name": "bob"})

        # 经典注入载荷：拼接实现会变成 `where id='x' or '1'='1'`，把两行都删掉
        table.delete({"id": "x' or '1'='1"})
        assert len(table.select_all()) == 2

        # 同一载荷用于查询时应当匹配不到任何记录
        assert table.select(condition={"id": "x' or '1'='1"}) == []
    finally:
        table.close()


def test_sql_injection_in_inserted_value_is_stored_verbatim(tmp_path):
    """带引号的字段值要原样入库，既不能改写 SQL，也不能被悄悄吃掉引号。"""
    from fundata.tables_bak.core import SqliteTable

    table = SqliteTable(
        db_path=str(tmp_path / "quote.db"), table_name="demo", columns=["id", "name"]
    )
    try:
        table.execute("create table demo (id varchar(50), name varchar(50))")
        table.insert({"id": "1", "name": "O'Brien"})
        assert table.select_all() == [{"id": "1", "name": "O'Brien"}]
    finally:
        table.close()


def test_download_name_is_parameterised(tmp_path):
    """download() 的数据集名来自调用方，注入载荷只能当普通名字，查不到就返回 None。"""
    dataset = _index_with_local_source(tmp_path, "demo", "demo.txt", "hello\n")
    try:
        assert dataset.download("' or '1'='1", path_root=str(tmp_path)) is None
        # 原表数据未被破坏
        assert len(dataset.select_all()) == 1
    finally:
        dataset.close()


def test_illegal_identifier_is_rejected():
    """表名/字段名无法参数化，只能靠白名单挡住，非法标识符必须直接拒绝。"""
    from fundata.exceptions import TableConfigError
    from fundata.tables_bak.core import BaseTable

    with pytest.raises(TableConfigError):
        BaseTable(table_name="demo; drop table users")
    with pytest.raises(TableConfigError):
        BaseTable(table_name="demo", columns=["id", "name); drop table users --"])


def test_empty_condition_does_not_wipe_table(tmp_path):
    """空条件会被拒绝，不能退化成全表删除，也不能生成 `where ` 这种坏 SQL。"""
    from fundata.exceptions import TableConfigError
    from fundata.tables_bak.core import SqliteTable

    table = SqliteTable(
        db_path=str(tmp_path / "empty.db"), table_name="demo", columns=["id", "name"]
    )
    try:
        table.execute("create table demo (id varchar(50), name varchar(50))")
        table.insert({"id": "1", "name": "alice"})

        with pytest.raises(TableConfigError):
            table.delete({})
        with pytest.raises(TableConfigError):
            table.delete({"not_a_column": "x"})
        assert len(table.select_all()) == 1

        # 显式传 None 才是「作用于全表」
        table.delete(None)
        assert table.select_all() == []
    finally:
        table.close()


def test_count_without_condition_counts_all_rows(tmp_path):
    """count() 不传条件时统计全表，而不是生成 `where ` 让 SQL 直接报错。"""
    from fundata.tables_bak.core import SqliteTable

    table = SqliteTable(
        db_path=str(tmp_path / "count.db"), table_name="demo", columns=["id", "name"]
    )
    try:
        table.execute("create table demo (id varchar(50), name varchar(50))")
        table.insert({"id": "1", "name": "alice"})
        table.insert({"id": "2", "name": "bob"})
        assert table.count() == 2
        assert table.count({"name": "alice"}) == 1
    finally:
        table.close()


def test_update_or_insert_requires_condition(tmp_path):
    """update_or_insert 不给条件时报领域异常，而不是 AttributeError。"""
    from fundata.exceptions import TableConfigError
    from fundata.tables_bak.core import SqliteTable

    table = SqliteTable(
        db_path=str(tmp_path / "upsert.db"), table_name="demo", columns=["id", "name"]
    )
    try:
        table.execute(
            "create table demo (id varchar(50) primary key, name varchar(50))"
        )
        with pytest.raises(TableConfigError):
            table.update_or_insert({"id": "1", "name": "alice"})

        table.update_or_insert({"id": "1", "name": "alice"}, condition={"id": "1"})
        assert table.select_all() == [{"id": "1", "name": "alice"}]
        table.update_or_insert({"id": "1", "name": "bob"}, condition={"id": "1"})
        assert table.select_all() == [{"id": "1", "name": "bob"}]
    finally:
        table.close()


def test_insert_list_binds_columns_explicitly(tmp_path):
    """批量插入显式列出字段名，字段顺序与建表顺序不同也不会错位。"""
    from fundata.tables_bak.core import SqliteTable

    table = SqliteTable(
        db_path=str(tmp_path / "bulk.db"), table_name="demo", columns=["name", "id"]
    )
    try:
        table.execute("create table demo (id varchar(50), name varchar(50))")
        table.insert_list([{"id": "1", "name": "alice"}, {"id": "2", "name": "bob"}])
        # columns 顺序是 (name, id)，建表顺序是 (id, name)。不写字段名的
        # `insert into demo values (?,?)` 会把 name 写进 id 列。
        raw = table.execute("select id, name from demo order by id").fetchall()
        assert raw == [("1", "alice"), ("2", "bob")]
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
