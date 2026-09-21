import os

from farlog import getLogger
from funshell import run_shell_list
from fundata.manage import DatasetManage, insert_library

logger = getLogger(__name__)


def run1():
    insert_library()


def run2():
    data = DatasetManage()
    cmd = "cp -r {} {}".format(
        data.db_path, os.path.abspath("./../fundata/manage/dataset.db")
    )
    logger.info(cmd)
    run_shell_list(
        ["cp", "-r", data.db_path, os.path.abspath("./../fundata/manage/dataset.db")]
    )
    sources = data.select_pd()

    lines = ["|序号|分类|名称|描述|官网下载|蓝奏下载|", "|:-:|:-:|:-:|:-:|:-:|:-:|"]
    for i, source in enumerate(sources.to_dict(orient="records")):
        source = data.decode(source)
        line = "|{i}|{category}|{name}|{describe}|{source_url}|{lanzou_url}|".format(
            i=i,
            name=source["name"],
            category=source["category"],
            describe=source["describe"],
            source_url="[官网链接]({})".format(source["urls"].get("source"))
            if "source" in source["urls"].keys()
            else "暂无",
            lanzou_url="[蓝奏链接]({})".format(source["urls"].get("lanzou"))
            if "lanzou" in source["urls"].keys()
            else "暂无",
        )
        lines.append(line)

    text = open("README.md", "r").read()
    text = text.replace("$dataset_table$", "\n".join(lines))
    open("../README.md", "w").write(text)


run1()
run2()
