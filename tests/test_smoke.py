"""Lightweight smoke tests for the ``fundata`` package.

Background / import-path notes
-------------------------------
The published distribution name and the importable top-level package
both use the *current* name: ``import fundata`` (NOT the legacy
``notedata`` name).

As of farfarfun/todo-list#154, ``fundata.work``, ``fundata.manage`` and
``fundata.tables_bak`` no longer reach for the dead ``notetool`` /
``notedrive`` package names -- they now use a local ``fundata._util``
helper, a real current dependency (``funshell``, ``funutil``), or (for
``manage.core.DatasetManage.download``'s lanzou branch, which had no
drop-in replacement in the current ``fundrive`` API) raise
``NotImplementedError`` instead of failing at import time.

``fundata.dataset`` is a separate, still-unresolved case: on top of the
now-fixed ``notedata.manage`` self-import, ``dataset/datas.py`` also
imports ``demjson`` / ``tensorflow`` / ``notekeras`` directly, none of
which are declared dependencies or installed here. That's a distinct,
pre-existing problem (missing/dead heavy ML deps) unrelated to the
note*->fun* renames #154 covers, so ``fundata.dataset`` still cannot be
imported -- documented and skipped below rather than faked as passing.
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
    creation via fundata._util.exist_and_create (no more notetool stub
    needed post-#154)."""
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

    NOTE: `delete()` is intentionally NOT exercised for correctness here;
    see `test_tables_bak_delete_condition_bug` below for a real bug found
    in that method (reported, not fixed, per audit scope).
    """
    from fundata.tables_bak.core import SqliteTable

    db_path = tmp_path / "smoke.db"
    table = SqliteTable(
        db_path=str(db_path), table_name="demo", columns=["id", "name"]
    )
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


def test_tables_bak_delete_condition_bug(tmp_path):
    """Found a real bug while smoke testing: `BaseTable.delete()` with a
    dict `condition` builds its SQL as::

        "delete from {} where {}".format(self.table_name, self._condition2equal(condition))

    `_condition2equal()` returns a *list* of `"col='value'"` strings (as
    `update()` correctly does too), but unlike `update()`, `delete()`
    never joins that list with `" and "` before formatting it into the
    SQL string. This produces invalid SQL such as::

        delete from demo where ["id='1'"]

    `SqliteTable.execute()` swallows the resulting sqlite3 error
    internally (prints it and returns None), so `delete(condition=dict)`
    silently does nothing instead of deleting the row or raising.

    This is a pre-existing business-logic bug in `fundata`'s own source
    (not something introduced by this test suite, and not something this
    test suite fixes -- out of scope). Skipping the correctness assertion
    and documenting it here instead of asserting the (currently broken)
    behaviour as if it were correct.
    """
    pytest.skip(
        "已发现但未修复的源码问题：BaseTable.delete() 在 condition 为 dict 时，"
        "未像 update() 一样对 _condition2equal() 返回的 list 做 ' and '.join()，"
        "导致拼出的 SQL 形如 \"delete from demo where [\\\"id='1'\\\"]\"，"
        "而 SqliteTable.execute() 会吞掉这个 sqlite3 语法错误并静默返回 None，"
        "因此 delete(dict 条件) 实际上什么都不会删除。按审计范围要求不修复源码，"
        "此处跳过并记录该发现。"
    )


def test_manage_dataset_manage_smoke(tmp_path):
    """fundata.manage.DatasetManage now imports cleanly post-#154 (subclasses
    the local fundata.tables_bak.core.SqliteTable instead of the dead
    notetool.database.SqliteTable). Exercise real CRUD against a throwaway
    sqlite db; the lanzou-download branch still can't be smoke tested
    (needs real credentials/network) so it's left untouched here."""
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


def test_manage_lanzou_download_not_implemented(tmp_path):
    """The lanzou branch of DatasetManage.download() has no drop-in
    replacement for the removed notedrive.lanzou.download free function
    (current fundrive.drives.lanzou.LanZouDrive is class-based and needs
    an authenticated instance) -- it now raises NotImplementedError
    instead of ImportError-ing the whole module. See #154.

    Note: DatasetManage.decode() does `json.loads(json.loads(urls))` --
    i.e. it expects `urls` to be *double* JSON-encoded -- while
    `encode()` (used by `insert()`) only encodes it *once*. That
    asymmetry is a separate, pre-existing bug (same flavour as the
    `delete()` condition bug above); this test works around it by
    storing a double-encoded value directly so `decode()` succeeds and
    execution actually reaches the lanzou branch under test.
    """
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


def test_import_dataset_submodule_requires_unavailable_deps():
    """fundata.dataset (core.py / datas.py / images.py) no longer imports the
    dead `notedata` name (fixed in #154 -- now a local `fundata.manage`
    self-import), but `dataset/datas.py` separately imports tensorflow /
    notekeras / demjson / scikit-learn directly, none of which are declared
    dependencies or installed here. This is an unrelated, pre-existing
    problem (missing heavy ML deps), not a note*->fun* naming issue, so it's
    out of scope for #154. Reported as a finding instead of faking a pass.
    """
    pytest.skip(
        "notedata/notetool/notedrive 引用已在 #154 修复（改为本地 fundata.manage 自引用 / "
        "funshell / funutil / fundata._util），但 fundata.dataset 内部 "
        "dataset/datas.py 仍直接 import demjson / tensorflow / notekeras / scikit-learn，"
        "这几个都不是本仓库声明的依赖，也未安装，属于与 note*->fun* 改名无关的另一类遗留问题"
        "（缺失/已废弃的重型 ML 依赖），不在 #154 范围内，已作为新发现记录，未修复。"
    )
