"""使用 fundrive 操作蓝奏云的示例。"""

from farlog import getLogger
from fundrive import get_drive

logger = getLogger(__name__)


def drive():
    """创建已登录的蓝奏云驱动；凭据由 fundrive/funsecret 读取。"""
    result = get_drive("lanzou")
    result.ignore_limit()
    result.login()
    return result


def example1():
    """验证蓝奏云登录状态。"""
    return drive()


def example2():
    """上传一个文件。"""
    file_path = "/Users/liangtaoniu/workspace/MyDiary/tmp/weights/yolov3.weight"
    return drive().upload_file(file_path=file_path, fid="2184164")


def example3():
    """按分享链接解析文件并下载到本地。"""
    result = drive().get_file_list(url="https://wws.lanzous.com/izZmlfjulvg")
    if not result:
        raise FileNotFoundError("分享链接中没有文件")
    return drive().download_file(result[0].fid, save_dir="./download/test")


def example4():
    """上传模型文件。"""
    res = None
    # downer.upload_file('/Users/liangtaoniu/workspace/MyDiary/tmp/models/yolo/configs/yolov3.h5', folder_id=2129808)
    # downer.upload_file('/Users/liangtaoniu/workspace/MyDiary/tmp/models/yolo/configs/yolov3.weights', folder_id=2129808)
    # downer.upload_file('/Users/liangtaoniu/workspace/dataset/models/yolov4.weights', folder_id=2129808)
    # res = downer.upload_file('/Users/liangtaoniu/workspace/dataset/models/annotations_trainval2017.zip',
    #                         folder_id=2160967)
    # res = downer.upload_file('/Users/liangtaoniu/workspace/dataset/models/val2017.zip', folder_id=2160967)

    # res = downer.upload_file('/Users/liangtaoniu/tmp/dataset/movielens/ml-100k.zip', folder_id=2184164)
    # res = downer.upload_file('/Users/liangtaoniu/tmp/dataset/movielens/ml-1m.zip', folder_id=2184164)
    # res = downer.upload_file('/Users/liangtaoniu/tmp/dataset/movielens/ml-10m.zip', folder_id=2184164)
    # res = downer.upload_file('/Users/liangtaoniu/tmp/dataset/movielens/ml-20m.zip', folder_id=2184164)

    # res = downer.upload_file('/Users/liangtaoniu/workspace/dataset/models/ml-25m.zip', folder_id=2184164)
    # res = downer.upload_file('/Users/liangtaoniu/workspace/dataset/models/train_data.csv', folder_id=2214573)
    res = drive().upload_file(
        "/Users/liangtaoniu/workspace/dataset/models/label_file.csv", folder_id=2214573
    )

    return res


def example5():
    """列出目录文件。"""
    return drive().get_dir_list(fid=2184164)
