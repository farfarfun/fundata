"""Lightweight smoke tests for the ``fundata`` package.

Background / import-path notes
-------------------------------
The published distribution name and the importable top-level package
both use the *current* name: ``import fundata`` (NOT the legacy
``notedata`` name). This was verified against ``pyproject.toml``
(``name = "fundata"``) and ``src/fundata/__init__.py``.

However, several internal submodules still reach for the OLD ``note*``
package names instead of the local ``fundata`` package, and those old
packages are no longer reliably installable:

* ``fundata.work``, ``fundata.manage``, ``fundata.tables_bak`` and
  ``fundata.dataset`` all import from ``notetool`` (e.g.
  ``from notetool.tool.path import exist_and_create``). ``notetool``
  does not exist on PyPI at all (``https://pypi.org/pypi/notetool/json``
  -> 404); the organisation appears to have renamed it to ``funtool``,
  but ``fundata``'s source was never updated to match. This looks like
  the same "incomplete rename" pattern flagged for this repo.
* ``fundata.manage`` also imports ``notedrive.lanzou``. ``notedrive``
  does exist on PyPI, but it in turn depends on ``demjson``, which fails
  to build on modern setuptools/Python (``error in demjson setup
  command: use_2to3 is invalid``), so ``notedrive`` cannot actually be
  installed in a fresh environment either.
* ``fundata.dataset`` additionally imports ``from notedata.manage import
  DatasetManage`` (``dataset/core.py``, ``dataset/datas.py``,
  ``dataset/images.py``) instead of the local ``fundata.manage``.
  ``notedata`` is still separately published on PyPI, but its own
  metadata declares a hard dependency on the (nonexistent) ``notetool``,
  so it can never be installed either. ``fundata.dataset`` also needs
  tensorflow / notekeras / demjson / scikit-learn.
* The ``example/`` scripts in this repo (``example/run.py``,
  ``example/dataset.py``) literally import ``notedata`` directly,
  which corroborates that this is a pre-existing, real bug rather than
  something introduced by this test suite.

None of this is fixed here (out of scope for a smoke-test-only change;
would require a source-level rename/refactor). Submodules that cannot be
meaningfully imported/exercised without real network access or
non-installable third-party packages are explicitly skipped below with
the reason spelled out, rather than silently omitted or faked as
passing.

For ``fundata.work`` and ``fundata.tables_bak`` the *only* missing piece
is a small filesystem/logging helper from ``notetool``
(``exist_and_create``, ``log``) that this test never needs to actually
touch the filesystem/network for, so it is stubbed out via
``sys.modules`` (same spirit as mocking a filesystem call) so the real
``fundata`` logic underneath can still be genuinely smoke tested.
"""

import logging
import sys
import types

import pytest


def _fake_module(name):
    return types.ModuleType(name)


def test_import_top_level_package():
    """`import fundata` must succeed with zero optional dependencies."""
    import fundata

    assert fundata.__name__ == "fundata"


def test_import_paths_submodule():
    """`fundata.paths` is a real, currently-empty submodule; must import cleanly."""
    import fundata.paths  # noqa: F401


def test_work_app_smoke(monkeypatch):
    """fundata.work.WorkApp: construct + path helpers, with the filesystem
    creation call (`notetool.tool.path.exist_and_create`) stubbed out so
    no real directories are touched and the missing `notetool` package
    doesn't block the import.
    """
    from unittest.mock import MagicMock

    fake_notetool = _fake_module("notetool")
    fake_notetool_tool = _fake_module("notetool.tool")
    fake_notetool_tool_path = _fake_module("notetool.tool.path")
    exist_and_create = MagicMock(name="exist_and_create")
    fake_notetool_tool_path.exist_and_create = exist_and_create
    fake_notetool_tool.path = fake_notetool_tool_path
    fake_notetool.tool = fake_notetool_tool

    monkeypatch.setitem(sys.modules, "notetool", fake_notetool)
    monkeypatch.setitem(sys.modules, "notetool.tool", fake_notetool_tool)
    monkeypatch.setitem(sys.modules, "notetool.tool.path", fake_notetool_tool_path)
    # fundata.work / fundata.work.core may already be cached from a
    # previous (failed) import attempt in this session; drop them so the
    # stub above is actually used.
    monkeypatch.delitem(sys.modules, "fundata.work", raising=False)
    monkeypatch.delitem(sys.modules, "fundata.work.core", raising=False)

    import fundata.work as work

    app = work.WorkApp(app_name="smoke-test-app", dir_app="/tmp/fundata-smoke-app")
    assert app.dir_db == "/tmp/fundata-smoke-app/databases"
    assert app.dir_log == "/tmp/fundata-smoke-app/logs"
    assert app.db_file("data.db") == "/tmp/fundata-smoke-app/databases/data.db"
    assert app.log_file("info.log") == "/tmp/fundata-smoke-app/logs/info.log"
    assert app.common_file("temp.txt") == "/tmp/fundata-smoke-app/common/temp.txt"

    # create() should delegate directory creation to the (stubbed)
    # filesystem helper rather than doing raw os calls itself.
    app.create()
    assert exist_and_create.call_count == 4

    # module-level convenience functions
    assert work.db_file(app_name="smoke-test-app", file_name="d.db").endswith(
        "databases/d.db"
    )
    assert work.log_file(app_name="smoke-test-app", file_name="l.log").endswith(
        "logs/l.log"
    )


def test_tables_bak_base_table_smoke(monkeypatch):
    """fundata.tables_bak.BaseTable: pure SQL-string-building logic, with
    only the `notetool.tool.log` logging helper stubbed out (a real
    `logging.getLogger` is used, so behaviour stays faithful) to work
    around the missing `notetool` package.
    """
    fake_notetool = _fake_module("notetool")
    fake_notetool_tool = _fake_module("notetool.tool")
    fake_notetool_tool.log = lambda name: logging.getLogger(name)
    fake_notetool.tool = fake_notetool_tool

    monkeypatch.setitem(sys.modules, "notetool", fake_notetool)
    monkeypatch.setitem(sys.modules, "notetool.tool", fake_notetool_tool)
    monkeypatch.delitem(sys.modules, "fundata.tables_bak", raising=False)
    monkeypatch.delitem(sys.modules, "fundata.tables_bak.core", raising=False)

    from fundata.tables_bak.core import BaseTable

    table = BaseTable(table_name="demo", columns=["id", "name"])
    assert table.table_name == "demo"
    assert isinstance(table.logger, logging.Logger)

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


def test_tables_bak_sqlite_table_crud_smoke(tmp_path, monkeypatch):
    """fundata.tables_bak.SqliteTable against a throwaway local sqlite
    file under pytest's tmp_path (no network, no credentials, no shared
    state) -- exercises real insert/select/update logic.

    NOTE: `delete()` is intentionally NOT exercised for correctness here;
    see `test_tables_bak_delete_condition_bug` below for a real bug found
    in that method (reported, not fixed, per audit scope).
    """
    fake_notetool = _fake_module("notetool")
    fake_notetool_tool = _fake_module("notetool.tool")
    fake_notetool_tool.log = lambda name: logging.getLogger(name)
    fake_notetool.tool = fake_notetool_tool

    monkeypatch.setitem(sys.modules, "notetool", fake_notetool)
    monkeypatch.setitem(sys.modules, "notetool.tool", fake_notetool_tool)
    monkeypatch.delitem(sys.modules, "fundata.tables_bak", raising=False)
    monkeypatch.delitem(sys.modules, "fundata.tables_bak.core", raising=False)

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


def test_tables_bak_delete_condition_bug(tmp_path, monkeypatch):
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


def test_import_manage_submodule_requires_unavailable_deps():
    """fundata.manage.DatasetManage subclasses `notetool.database.SqliteTable`
    and its `download()` method unconditionally hits the network (lanzou
    file hosting) via `notedrive.lanzou.download`.

    - `notetool` does not exist on PyPI (404), so `notetool.database`
      cannot be installed/imported at all.
    - `notedrive` exists on PyPI, but its own dependency `demjson` fails
      to build on this environment (`use_2to3 is invalid`), so it cannot
      be installed either.

    With both of the class's dependencies unavailable, and its own
    behaviour requiring real network access, this cannot be reasonably
    smoke tested even with mocking (doing so would require reimplementing
    a third-party base class from scratch). Skipping per audit guidance
    instead of faking a pass; reported as a finding.
    """
    pytest.skip(
        "需要真实凭据/网络，跳过：fundata.manage.DatasetManage 依赖 PyPI 上已不存在的 "
        "notetool 包（notetool.database.SqliteTable）以及构建失败的 notedrive"
        "（其依赖 demjson 在现代 setuptools/Python 下报错 'use_2to3 is invalid' 无法构建），"
        "且 download() 方法会真实请求蓝奏云下载文件。在不重新实现第三方基类的前提下，"
        "无法对其进行有意义的 mock 冒烟测试，已作为发现问题记录，未修复源码。"
    )


def test_import_dataset_submodule_requires_unavailable_deps():
    """fundata.dataset (core.py / datas.py / images.py) imports `from
    notedata.manage import DatasetManage` -- an OLD package name -- instead
    of the local `fundata.manage`, plus tensorflow / notekeras / demjson /
    scikit-learn.

    - `notedata` is still separately published on PyPI, but its own
      metadata declares a hard dependency on the nonexistent `notetool`,
      so `notedata` itself can never be installed.
    - `demjson` (needed directly by `dataset/datas.py`) fails to build on
      modern setuptools/Python.
    - tensorflow is a very heavy dependency that would only serve to
      reach code that is already blocked by the two points above.

    This looks like the same incomplete "notedata -> fundata" rename
    referenced in the tracking issue: `fundata`'s own dataset submodule
    never got its internal import updated from `notedata.manage` to the
    local `fundata.manage`. Confirmed further by `example/run.py` and
    `example/dataset.py` in this repo, which still import `notedata`
    directly. Not fixed here (source refactor, out of scope for this
    test-only change); reported as a finding instead of faking a pass.
    """
    pytest.skip(
        "需要真实凭据/网络，跳过：fundata.dataset 内部仍从旧包名 notedata 导入 "
        "(`from notedata.manage import DatasetManage`)，而不是本地的 fundata.manage，"
        "疑似 notedata -> fundata 改名未完全同步的遗留问题（example/run.py、example/dataset.py "
        "中同样直接 import notedata，可佐证）；同时 notedata 自身在 PyPI 上声明依赖已不存在的 "
        "notetool 而无法安装，demjson 在现代环境下构建失败，tensorflow 体积过大且无法绕开上述阻塞，"
        "已作为发现问题记录，未修复源码。"
    )
