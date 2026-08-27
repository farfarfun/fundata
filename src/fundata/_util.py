"""Small file-path helpers that used to live in the now-defunct ``notetool``
package. ``notetool`` was renamed to ``funtool`` (PyPI: ``farfuntool``), but
``funtool.tool`` itself still imports the dead ``notetool`` name (a separate,
unfixed bug), so it cannot be depended on here. These are minimal local
reimplementations of just the functions fundata actually needs.
"""

import os


def path_parse(path):
    if path is None:
        return path
    path = os.path.expanduser(path)
    if not path.startswith("/"):
        return os.path.join(os.getcwd(), path)
    return path


def exist_and_create(file_dir):
    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return file_dir


def exists_file(file_path, mkdir=False):
    if mkdir:
        exist_and_create(os.path.dirname(file_path))
    return os.path.exists(file_path)
