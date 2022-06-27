import pandas as pd
from .datas import ElectronicsData
from notedata.manage import DatasetManage
from notetool.tool import log

logger = log(__name__)


def get_electronics(dataset: DatasetManage = None):
    electronic = ElectronicsData(dataset=dataset, data_path='./download/')
    electronic.init_data()


def get_movielens(dataset: DatasetManage = None):
    dataset.download('movielens-100k', overwrite=False)
    dataset.download('movielens-1m', overwrite=False)
    dataset.download('movielens-10m', overwrite=False)
    dataset.download('movielens-20m', overwrite=False)
    dataset.download('movielens-25m', overwrite=False)
    # os.system('cd ' + file_path(data.path) + ' && unzip ' + file_name(data.path))


def get_adult_data(dataset: DatasetManage = None):
    data_train = dataset.download('adult-train', overwrite=False)
    data_test = dataset.download('adult-test', overwrite=False)

    print(data_test)
    train_data = pd.read_table(data_train.path, header=None, delimiter=',')
    test_data = pd.read_table(
        data_test.path, header=None, delimiter=',', error_bad_lines=False)

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


def get_porto_seguro_data(dataset: DatasetManage = None):
    dataset.download('porto-seguro-train')
    dataset.download('porto-seguro-test')


def get_bitly_usagov_data(dataset: DatasetManage = None):
    dataset.download('bitly-usagov')
