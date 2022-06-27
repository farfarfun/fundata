import json

from tqdm import tqdm

from .core import DatasetManage


def check():
    dataset = DatasetManage()
    dataset.create()

    data = dataset.select_all()
    d1 = json.loads(json.loads(data[0][3]))
    print(d1)
    print(d1['source'])


def insert_library():
    lines = []
    # iris
    lines.extend([
        {
            'name': 'iris',
            'category': 'dataset',
            'describe': 'iris数据集',
            'urls': {
                'source': 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
            },
            "path": 'iris/iris.data'}])

    # criteo
    lines.extend([
        {
            'name': 'criteo-sample',
            'category': 'dataset',
            'describe': 'criteo_sample数据集',
            'urls': {
                'source': 'https://raw.githubusercontent.com/shenweichen/DeepCTR/master/examples/criteo_sample.txt',
                "lanzou": "https://wws.lanzous.com/ihLhrhejkxi"
            },
            "path": 'criteo/criteo_sample.txt'},
        {
            'name': 'criteo-kaggle',
            'category': 'dataset',
            'describe': 'criteo-kaggle数据集',
            'urls': {
                'source': 'https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz',
                "lanzou": "https://wws.lanzous.com/b01hqh97i"
            },
            "path": 'criteo/dac.tar.gz'}
    ])

    # electronics
    lines.extend([
        {
            'name': 'electronics-reviews',
            'category': 'dataset',
            'describe': 'Amazon评论数据',
            'urls': {
                "source": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz",
                "lanzou": "https://wws.lanzous.com/b01hkzfuj"
            },
            'path': 'electronics/reviews_Electronics_5.json.gz'},
        {
            'name': 'electronics-meta',
            'category': 'dataset',
            'describe': 'Amazon评论数据',
            'urls': {
                "source": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz",
                "lanzou": "https://wws.lanzous.com/b01hqeora"
            },
            'path': 'electronics/meta_Electronics.json.gz'}])

    # movielens
    lines.extend([
        {
            'category': 'dataset',
            'name': 'movielens-100k',
            'urls': {
                "source": "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
                "lanzou": "https://wws.lanzous.com/iyykCfbi64j"
            },
            'path': 'movielens/ml-100k.zip',
            'describe': '包含用户对电影的评级数据、电影元数据信息和用户属性信息',
            'md5': '0e33842e24a9c977be4e0107933c0723'
        },
        {
            'category': 'dataset',
            'name': 'movielens-1m',
            'urls': {
                "source": "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
                "lanzou": "https://wws.lanzous.com/ihoSUfbi65a"
            },
            'path': 'movielens/ml-1m.zip',
            'describe': '包含用户对电影的评级数据、电影元数据信息和用户属性信息',
            'md5': 'c4d9eecfca2ab87c1945afe126590906'
        },
        {
            'category': 'dataset',
            'name': 'movielens-10m',
            'urls': {
                "source": "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
                "lanzou": "https://wws.lanzous.com/iXvEmfbi6di"
            },
            'path': 'movielens/ml-10m.zip',
            'describe': '包含用户对电影的评级数据、电影元数据信息和用户属性信息',
            'md5': 'ce571fd55effeba0271552578f2648bd'
        },
        {
            'category': 'dataset',
            'name': 'movielens-20m',
            'describe': '包含用户对电影的评级数据、电影元数据信息和用户属性信息',
            'urls': {
                "source": "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
                "lanzou": "https://wws.lanzous.com/b01hkt17g"
            },
            'path': 'movielens/ml-20m.zip',
            'md5': 'cd245b17a1ae2cc31bb14903e1204af3'
        },
        {
            'category': 'dataset',
            'name': 'movielens-25m',
            'urls': {
                "source": "http://files.grouplens.org/datasets/movielens/ml-25m.zip",
                "lanzou": "https://wws.lanzous.com/b01hkt24j"
            },
            'path': 'movielens/ml-25m.zip',
            'describe': '包含用户对电影的评级数据、电影元数据信息和用户属性信息',
            'md5': '6b51fb2759a8657d3bfcbfc42b592ada'
        }])

    # adult_data
    lines.extend([
        {
            'name': 'adult-train', 'category': 'dataset',
            'urls': {
                'source': 'https://raw.githubusercontent.com/1007530194/data/master/recommendation/data/adult.data.txt'
            },
            "path": 'adult-data/adult.train.txt',
            'describe': '',
        }, {
            'name': 'adult-test', 'category': 'dataset',
            'urls': {
                'source': 'https://raw.githubusercontent.com/1007530194/data/master/recommendation/data/adult.test.txt'
            },
            "path": 'adult-data/adult.test.txt',
            'describe': '',
        }
    ])

    # porto_seguro
    lines.extend([
        {
            'name': 'porto-seguro-train', 'category': 'dataset',
            'urls': {
                'source': 'https://raw.githubusercontent.com/1007530194/data/master/recommendation/data/porto_seguro_train.csv'
            },
            "path": 'porto-seguro/porto_seguro_train.csv',
            'describe': '',
        }, {
            'name': 'porto-seguro-test', 'category': 'dataset',
            'urls': {
                'source': 'https://raw.githubusercontent.com/1007530194/data/master/recommendation/data/porto_seguro_test.csv'
            },
            "path": 'porto-seguro/porto_seguro_test.csv',
            'describe': '',
        }
    ])

    # porto_seguro
    lines.extend([
        {
            'name': 'bitly-usagov', 'category': 'dataset',
            'urls': {
                'source': 'https://raw.githubusercontent.com/1007530194/data/master/datasets/bitly_usagov/example.txt'
            },
            "path": 'bitly-usagov/bitly_usagov_train.csv',
            'describe': '',
        }
    ])

    # coco
    lines.extend([
        {
            'name': 'coco-val2017',
            'category': 'dataset',
            'urls': {
                'source': 'http://images.cocodataset.org/zips/val2017.zip',
                'lanzou': 'https://wws.lanzous.com/b01hkb8fi'
            },
            "path": 'coco/val2017.zip',
            'describe': '大型图像数据集, 用于对象检测、分割、人体关键点检测、语义分割和字幕生成',
        },
        {
            'name': 'coco-annotations_trainval2017',
            'category': 'dataset',
            'urls': {
                'source': "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
                'lanzou': 'https://wws.lanzous.com/b01hkb86j'
            },
            "path": 'coco/annotations_trainval2017.zip',
            'describe': '大型图像数据集, 用于对象检测、分割、人体关键点检测、语义分割和字幕生成',
        }
    ])

    # model-yolo
    lines.extend([{
        'name': 'yolov3.weight',
        'category': 'model',
        'urls': {
            'lanzou': 'https://wws.lanzous.com/b01hjn3ih'
        },
        "path": 'model/yolo/',
        'describe': 'yolov3模型的权重',
    }, {
        'name': 'yolov3.h5',
        'category': 'model',
        'urls': {
            'lanzou': 'https://wws.lanzous.com/b01hjn3aj'
        },
        "path": 'model/yolo/',
        'describe': 'yolov3模型的权重',
    }, {
        'name': 'yolov4.weight',
        'category': 'model',
        'urls': {
            'lanzou': 'https://wws.lanzous.com/b01hjn3yd'
        },
        "path": 'model/yolo/',
        'describe': 'yolov4模型的权重',
    }, {
        'name': 'yolov4-416.h5',
        'category': 'model',
        'urls': {
            'lanzou': 'https://wws.lanzous.com/b01hl9lej'
        },
        "path": 'model/yolo/',
        'describe': 'yolov4模型的权重',
    }])

    dataset = DatasetManage()
    dataset.create()
    for line in tqdm(lines):
        dataset.insert(line)
        dataset.update(line)
