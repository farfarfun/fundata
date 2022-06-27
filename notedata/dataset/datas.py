import os
import pickle
import random

import demjson
import numpy as np
import pandas as pd
import tensorflow as tf
from notedata.manage import DatasetManage
from notekeras.features.feature_parse import define_feature_json
from notetool.tool import exists_file, log
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

logger = log(__name__)


class DataSet:
    """
    """

    def __init__(self, dataset: DatasetManage = None, data_path='./download/'):
        """
        """
        # 源文件保存目录
        self.dataset = dataset or DatasetManage()
        self.path_root = data_path

    def download(self, mode=1):
        """
        """
        pass

    def preprocess(self, step=0):
        """
        """
        pass

    def build_dataset(self):
        """
        """
        pass


class ElectronicsData(DataSet):
    def __init__(self, *args, **kwargs):
        super(ElectronicsData, self).__init__(*args, **kwargs)

        # 源文件
        self.json_meta = self.path_root + '/electronics/meta_Electronics.json'
        self.json_reviews = self.path_root + '/electronics/reviews_Electronics_5.json'

        # 格式化文件
        self.pkl_meta = self.path_root + '/electronics/raw_data/meta.pkl'
        self.pkl_reviews = self.path_root + '/electronics/raw_data/reviews.pkl'

        # 结果文件
        self.pkl_remap = self.path_root + '/electronics/raw_data/remap.pkl'
        self.pkl_dataset = self.path_root + '/electronics/raw_data/dataset.pkl'

    def download_raw_0(self, overwrite=False):
        self.dataset.download('electronics-reviews',
                              overwrite=overwrite, path_root=self.path_root)
        self.dataset.download('electronics-meta',
                              overwrite=overwrite, path_root=self.path_root)

        logger.info("download done")
        logger.info("begin unzip file")

        os.system('cd ' + self.path_root +
                  '/electronics && gzip -d reviews_Electronics_5.json.gz')
        os.system('cd ' + self.path_root +
                  '/electronics && gzip -d meta_Electronics.json.gz')
        logger.info("unzip done")

    def convert_pd_1(self, overwrite=False):
        if exists_file(self.pkl_reviews, mkdir=True) and exists_file(self.pkl_meta, mkdir=True):
            return

        def to_df(file_path):
            with open(file_path, 'r') as fin:
                df = {}
                i = 0
                for line in fin:
                    df[i] = eval(line)
                    i += 1
                df = pd.DataFrame.from_dict(df, orient='index')
                return df

        reviews_df = to_df(self.json_reviews)
        with open(self.pkl_reviews, 'wb') as f:
            pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

        meta_df = to_df(self.json_meta)
        meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
        meta_df = meta_df.reset_index(drop=True)
        with open(self.pkl_meta, 'wb') as f:
            pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)

    def remap_id_2(self, overwrite=False):
        random.seed(1234)
        if exists_file(self.pkl_remap, mkdir=True):
            return

        # reviews
        reviews_df = pd.read_pickle(self.pkl_reviews)
        reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]
        # meta
        meta_df = pd.read_pickle(self.pkl_meta)
        meta_df = meta_df[['asin', 'categories']]
        # 类别只保留最后一个
        meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

        # with open(self.pkl_reviews, 'rb') as f:
        #     reviews_df = pickle.load(f)
        #     reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

        # with open(self.pkl_meta, 'rb') as f:
        #     meta_df = pickle.load(f)
        #     meta_df = meta_df[['asin', 'categories']]
        #     meta_df['categories'] = meta_df['categories'].map(
        #         lambda x: x[-1][-1])

        def build_map(df, col_name):
            """
            制作一个映射，键为列名，值为序列数字
            :param df: reviews_df / meta_df
            :param col_name: 列名
            :return: 字典，键
            """
            key = sorted(df[col_name].unique().tolist())
            m = dict(zip(key, range(len(key))))
            df[col_name] = df[col_name].map(lambda x: m[x])
            return m, key

        # meta_df文件的物品ID映射
        asin_map, asin_key = build_map(meta_df, 'asin')
        # meta_df文件物品种类映射
        cate_map, cate_key = build_map(meta_df, 'categories')
        # reviews_df文件的用户ID映射
        view_map, view_key = build_map(reviews_df, 'reviewerID')

        # user_count: 192403	item_count: 63001	cate_count: 801	example_count: 1689188
        user_count, item_count, cate_count, example_count = \
            len(view_map), len(asin_map), len(cate_map), reviews_df.shape[0]
        logger.info('user_count: %d\t item_count: %d\t cate_count: %d\t example_count: %d' %
                    (user_count, item_count, cate_count, example_count))

        # 按物品id排序，并重置索引
        meta_df = meta_df.sort_values('asin')
        meta_df = meta_df.reset_index(drop=True)

        # reviews_df文件物品id进行映射，并按照用户id、浏览时间进行排序，重置索引
        reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
        reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
        reviews_df = reviews_df.reset_index(drop=True)
        reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

        # 各个物品对应的类别
        cate_list = np.array(meta_df['categories'], dtype='int32')
        # cate_list = [meta_df['categories'][i] for i in range(len(asin_map))]
        # cate_list = np.array(cate_list, dtype=np.int32)

        # 保存所需数据为pkl文件
        with open(self.pkl_remap, 'wb') as f:
            pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)  # uid, iid
            pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump((user_count, item_count, cate_count, example_count),
                        f, pickle.HIGHEST_PROTOCOL)
            pickle.dump((asin_key, cate_key, view_key),
                        f, pickle.HIGHEST_PROTOCOL)

    def build_dataset_3(self, overwrite=False):
        random.seed(1234)
        if exists_file(self.pkl_dataset, mkdir=True):
            return

        with open(self.pkl_remap, 'rb') as f:
            reviews_df = pickle.load(f)
            cate_list = pickle.load(f)
            user_count, item_count, cate_count, example_count = pickle.load(f)

        train_set, test_set = [], []
        # 最大的序列长度
        max_sl = 0

        """
        生成训练集、测试集，每个用户所有浏览的物品（共n个）前n-1个为训练集（正样本），并生成相应的负样本，每个用户
        共有n-2个训练集（第1个无浏览历史），第n个作为测试集。
        故测试集共有192403个，即用户的数量。训练集共2608764个
        """
        for reviewerID, hist in reviews_df.groupby('reviewerID'):
            # 每个用户浏览过的物品，即为正样本
            pos_list = hist['asin'].tolist()
            max_sl = max(max_sl, len(pos_list))

            def gen_neg():
                neg = pos_list[0]
                while neg in pos_list:
                    neg = random.randint(0, item_count - 1)
                return neg

            # 正负样本比例1：1
            neg_list = [gen_neg() for i in range(len(pos_list))]

            for i in range(1, len(pos_list)):
                # 生成每一次的历史记录，即之前的浏览历史
                hist = pos_list[:i]
                sl = len(hist)
                if i != len(pos_list) - 1:
                    # 保存正负样本，格式：用户ID，正/负物品id，浏览历史，浏览历史长度，标签（1/0）
                    train_set.append((reviewerID, pos_list[i], hist, sl, 1))
                    train_set.append((reviewerID, neg_list[i], hist, sl, 0))
                else:
                    # 最后一次保存为测试集
                    label = (pos_list[i], neg_list[i])
                    test_set.append((reviewerID, hist, sl, label))

        # 打乱顺序
        random.shuffle(train_set)
        random.shuffle(test_set)

        assert len(test_set) == user_count

        # 写入dataset.pkl文件
        with open(self.pkl_dataset, 'wb') as f:
            pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump((user_count, item_count, cate_count,
                         max_sl), f, pickle.HIGHEST_PROTOCOL)

    def init_data(self, overwrite=False):
        self.download_raw_0(overwrite=overwrite)

        self.convert_pd_1(overwrite=overwrite)

        self.remap_id_2(overwrite=overwrite)

        self.build_dataset_3(overwrite=overwrite)

    def download(self, mode=1):
        self.download_raw_0(overwrite=False)

    def preprocess(self, step=0):
        if step == 0:
            self.convert_pd_1()
            self.remap_id_2()
            self.build_dataset_3()
        elif step == 1 or step == 'convert pd':
            self.convert_pd_1()
        elif step == 2 or step == 'remap id':
            self.remap_id_2()
        elif step == 3 or step == 'build dataset':
            self.build_dataset_3()


class CriteoDataBak(DataSet):
    def __init__(self, *args, **kwargs):
        super(CriteoDataBak, self).__init__(*args, **kwargs)
        self.criteo_sample = self.path_root+'/criteo/criteo_sample.txt'
        self.criteo_kaggle = self.path_root+'/criteo/criteo_sample.txt'
        self.criteo_kaggle_train = self.path_root+'/criteo/train.txt'
        self.criteo_kaggle_test = self.path_root+'/criteo/test.txt'
        self.sample_num = 1000000

    def download(self, mode=1):
        if mode == 1:
            self.dataset.download('criteo-sample', path_root=self.path_root)
        elif mode == 2:
            self.dataset.download('criteo-kaggle', path_root=self.path_root)
            os.system('cd ' + self.path_root +
                      '/criteo && tar -zxvf dac.tar.gz')

    def preprocess(self, step=0):
        pass

    def _sparseFeature(self, feat, feat_num, embed_dim=4):
        """
        create dictionary for sparse feature
        :param feat: feature name
        :param feat_num: the total number of sparse features that do not repeat
        :param embed_dim: embedding dimension
        :return:
        """
        return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

    def _denseFeature(self, feat):
        """
        create dictionary for dense feature
        :param feat: dense feature name
        :return:
        """
        return {'feat': feat}

    def _create_criteo_dataset(self, file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
        """
        a example about creating criteo dataset
        :param file: dataset's path
        :param embed_dim: the embedding dimension of sparse features
        :param read_part: whether to read part of it
        :param sample_num: the number of instances if read_part is True
        :param test_size: ratio of train dataset to test dataset
        :return: feature columns, train, test
        """
        names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
                 'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
                 'C23', 'C24', 'C25', 'C26']

        if read_part:
            data_df = pd.read_csv(file, sep='\t', iterator=True, header=None,
                                  names=names)
            data_df = data_df.get_chunk(sample_num)

        else:
            data_df = pd.read_csv(file, sep=',', header=None, names=names)

        sparse_features = ['C' + str(i) for i in range(1, 27)]
        dense_features = ['I' + str(i) for i in range(1, 14)]

        data_df[sparse_features] = data_df[sparse_features].fillna('-1')
        data_df[dense_features] = data_df[dense_features].fillna(0)

        for feat in sparse_features:
            le = LabelEncoder()
            data_df[feat] = le.fit_transform(data_df[feat])

        # ==============Feature Engineering===================

        dense_features = [
            feat for feat in data_df.columns if feat not in sparse_features + ['label']]

        mms = MinMaxScaler(feature_range=(0, 1))
        data_df[dense_features] = mms.fit_transform(data_df[dense_features])

        feature_columns = [[self._denseFeature(feat) for feat in dense_features]] + \
            [[self._sparseFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim)
              for feat in sparse_features]]

        train, test = train_test_split(data_df, test_size=test_size)

        train_X = [train[dense_features].values,
                   train[sparse_features].values.astype('int32')]
        train_y = train['label'].values.astype('int32')
        test_X = [test[dense_features].values,
                  test[sparse_features].values.astype('int32')]
        test_y = test['label'].values.astype('int32')

        return feature_columns, (train_X, train_y), (test_X, test_y)

    def build_dataset(self, mode=1):
        if mode == 1:
            return self._create_criteo_dataset(self.criteo_sample, read_part=False)
        elif mode == 2:
            return self._create_criteo_dataset(self.criteo_kaggle_train, read_part=True, sample_num=self.sample_num)


class CriteoData(DataSet):
    def __init__(self, *args, **kwargs):
        super(CriteoData, self).__init__(*args, **kwargs)
        self.criteo_sample = self.path_root + '/criteo/criteo_sample.txt'
        self.criteo_kaggle = self.path_root + '/criteo/criteo_sample.txt'
        self.criteo_kaggle_train = self.path_root + '/criteo/train.txt'
        self.criteo_kaggle_test = self.path_root + '/criteo/test.txt'

        self.feature_file = self.path_root + '/criteo/feature_layers.json'
        self.sample_num = 1000000

    def download(self, mode=1):
        if mode == 1:
            self.dataset.download('criteo-sample', path_root=self.path_root)
        elif mode == 2:
            self.dataset.download('criteo-kaggle', path_root=self.path_root)
            os.system('cd ' + self.path_root +
                      '/criteo && tar -zxvf dac.tar.gz')

    def _create_criteo_dataset(self, file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2, batch_size=128):
        """
        a example about creating criteo dataset
        :param file: dataset's path
        :param embed_dim: the embedding dimension of sparse features
        :param read_part: whether to read part of it
        :param sample_num: the number of instances if read_part is True
        :param test_size: ratio of train dataset to test dataset
        :return: feature columns, train, test
        """
        dense_features = ['I' + str(i) for i in range(1, 14)]
        sparse_features = ['C' + str(i) for i in range(1, 27)]
        names = ['label', *dense_features, *sparse_features]

        if read_part:
            data_df = pd.read_csv(
                file, sep='\t', iterator=True, header=None, names=names)
            data_df = data_df.get_chunk(sample_num)
        else:
            data_df = pd.read_csv(file, sep=',', header=None, names=names)

        data_df[dense_features] = data_df[dense_features].fillna(
            0.).astype('float32')
        data_df[sparse_features] = data_df[sparse_features].fillna('-1')

        for feat in sparse_features:
            le = LabelEncoder()
            data_df[feat] = le.fit_transform(data_df[feat])

        sparse_json_embedding, sparse_json_onehot = [], []
        for feat in sparse_features:
            vocabulary = [i for i in data_df[feat].unique()]
            sparse_json_embedding.append(define_feature_json(feat,
                                                             feature_type='CateEmbeddingColumn',
                                                             dimension=embed_dim,
                                                             share_name=feat + '-emb',
                                                             vocabulary_size=len(
                                                                 vocabulary),
                                                             dtype='int'))
            sparse_json_onehot.append(define_feature_json(feat,
                                                          feature_type='CateOneHotColumn',
                                                          dimension=embed_dim,
                                                          num_buckets=len(
                                                              vocabulary),
                                                          dtype='int'))

        mms = MinMaxScaler(feature_range=(0, 1))
        data_df[dense_features] = mms.fit_transform(data_df[dense_features])

        train, test = train_test_split(data_df, test_size=test_size)

        def df_to_dataset(dataframe, shuffle=True, batch_size=32):
            dataframe = dataframe.copy()
            labels = dataframe.pop('label')
            data = dataframe.to_dict(orient='list')
            ds = tf.data.Dataset.from_tensor_slices((data, labels))

            if shuffle:
                ds = ds.shuffle(buffer_size=len(dataframe))
            ds = ds.batch(batch_size)
            return ds

        train_d = df_to_dataset(train, batch_size=batch_size)
        test_d = df_to_dataset(test, batch_size=batch_size)

        feature_layers = {
            'num-layer': {
                "type": "single",
                "inputs": [define_feature_json(key=name, feature_type='NumericColumn', dtype='float32') for name
                           in dense_features]
            },
            'cate-embedding': {
                "type": "single",
                "inputs": sparse_json_embedding
            },
            'cate-onehot': {
                "type": "single",
                "inputs": sparse_json_onehot
            }
        }

        with open(self.feature_file, 'w') as writer:
            writer.write(demjson.encode(feature_layers))

        return feature_layers, train_d, test_d
        # return feature_layers, (train.to_dict(orient='list'), train[['label']].to_dict(orient='list')), (test.to_dict(orient='list'), test[['label']].to_dict(orient='list'))

    def build_dataset(self, mode=1, batch_size=1024):
        if mode == 1:
            return self._create_criteo_dataset(self.criteo_sample, read_part=False)
        elif mode == 2:
            return self._create_criteo_dataset(self.criteo_kaggle_train, read_part=True, sample_num=self.sample_num, batch_size=batch_size)
