import logging

from notedrive.lanzou import LanZouCloud, CodeDetail, download
from notedrive.lanzou.utils import logger

logger.setLevel(logging.DEBUG)

downer = LanZouCloud()
downer.ignore_limits()

downer.login_by_cookie()


def example1():
    print(downer.login_by_cookie() == CodeDetail.SUCCESS)


def example2():
    file_path = '/Users/liangtaoniu/workspace/MyDiary/tmp/weights/yolov3.weight'
    downer.upload_file(file_path=file_path)


def example3():
    print('download')
    download('https://wws.lanzous.com/izZmlfjulvg', dir_pwd='./download/test')

    # download('https://wws.lanzous.com/b01hjn3aj', dir_pwd='./download/lanzou')

    # download('https://wws.lanzous.com/b01hh63kf', dir_pwd='./download/lanzou')
    # downer.down_dir_by_url('https://wws.lanzous.com/b01hh2zve', dir_pwd='./download/lanzou')

    pass


def example4():
    print("upload")
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
    res = downer.upload_file('/Users/liangtaoniu/workspace/dataset/models/label_file.csv', folder_id=2214573)

    print(res)
    pass


def example5():
    print(downer.get_dir_list(folder_id=2184164))


# example1()
# example2()
example3()
# example4()
# example5()
# https://wws.lanzous.com/b01hjn3aj
# print(downer._session.cookies)
