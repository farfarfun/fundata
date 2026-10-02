import pandas as pd
from farlog import getLogger

from ..exceptions import DatasetNotFoundError
from ..manage import DatasetManage
from .datas import ElectronicsData

logger = getLogger(__name__)


def _get_dataset(dataset: DatasetManage | None) -> DatasetManage:
    """返回调用方传入的数据集管理器，未传入时创建默认实例。"""
    return dataset or DatasetManage()


def get_electronics(dataset: DatasetManage | None = None) -> None:
    """下载并处理 Amazon Electronics 数据集。"""
    electronic = ElectronicsData(dataset=dataset, data_path="./download/")
    electronic.init_data()


def get_movielens(dataset: DatasetManage | None = None) -> None:
    """下载 MovieLens 数据集。"""
    dataset = _get_dataset(dataset)
    dataset.download("movielens-100k", overwrite=False)
    dataset.download("movielens-1m", overwrite=False)
    dataset.download("movielens-10m", overwrite=False)
    dataset.download("movielens-20m", overwrite=False)
    dataset.download("movielens-25m", overwrite=False)
    # os.system('cd ' + file_path(data.path) + ' && unzip ' + file_name(data.path))


def get_adult_data(
    dataset: DatasetManage | None = None, path_root: str = "./download/"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """下载并读取 Adult 数据集。

    :param dataset: 数据集索引管理器，不传则创建默认实例
    :param path_root: 本地数据根目录
    :return: ``(训练集, 测试集)`` 两个 DataFrame
    :raises DatasetNotFoundError: ``adult-train``/``adult-test`` 不在索引中时抛出
    """
    dataset = _get_dataset(dataset)
    train_path = dataset.download("adult-train", overwrite=False, path_root=path_root)
    test_path = dataset.download("adult-test", overwrite=False, path_root=path_root)

    if train_path is None or test_path is None:
        raise DatasetNotFoundError(
            "adult-train/adult-test 不在数据集索引中，请先执行 insert_library()"
        )

    train_data = pd.read_table(train_path, header=None, delimiter=",")
    # UCI 原始 adult.test 第一行是 `|1x3 Cross validator` 注释行，不跳过的话 pandas
    # 会按它推断出只有 1 列，再把后面 15 列的正常行全部当成坏行丢掉。
    test_data = pd.read_table(
        test_path, header=None, delimiter=",", comment="|", on_bad_lines="skip"
    )
    logger.info(f"adult 数据: train={train_data.shape} test={test_data.shape}")
    return train_data, test_data

    # all_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
    #                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    #                'label', 'type']
    #
    # continus_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    # dummy_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
    #                  'native-country']
    #
    # train_data['type'] = 1
    # test_data['type'] = 2
    #
    # all_data = pd.concat([train_data, test_data], axis=0)
    # all_data.columns = all_columns
    #
    # all_data = pd.get_dummies(all_data, columns=dummy_columns)
    #
    # test_data = all_data[all_data['type'] == 2].drop(['type'], axis=1)
    # train_data = all_data[all_data['type'] == 1].drop(['type'], axis=1)
    #
    # train_data['label'] = train_data['label'].map(lambda x: 1 if x.strip() == '>50K' else 0)
    # test_data['label'] = test_data['label'].map(lambda x: 1 if str(x).strip() == '>50K.' else 0)
    #
    # for col in continus_columns:
    #     ss = StandardScaler()
    #     train_data[col] = ss.fit_transform(train_data[[col]].astype(np.float64))
    #     test_data[col] = ss.transform(test_data[[col]].astype(np.float64))
    #
    # train_y = train_data['label']
    # train_x = train_data.drop(['label'], axis=1)
    # test_y = test_data['label']
    # test_x = test_data.drop(['label'], axis=1)
    #
    # return train_x, train_y, test_x, test_y


def get_porto_seguro_data(dataset: DatasetManage | None = None) -> None:
    """下载 Porto Seguro 训练和测试数据。"""
    dataset = _get_dataset(dataset)
    dataset.download("porto-seguro-train")
    dataset.download("porto-seguro-test")


def get_bitly_usagov_data(dataset: DatasetManage | None = None) -> None:
    """下载 Bitly USA.gov 数据。"""
    dataset = _get_dataset(dataset)
    dataset.download("bitly-usagov")
