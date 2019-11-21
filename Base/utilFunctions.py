from sklearn import preprocessing, model_selection
import numpy as np
import scipy.sparse as sps
import pandas as pd
import random


# get users and items lists from urm_file
def get_urm_lists(urm_file):
    urm = pd.read_csv(urm_file)
    users_list = np.asarray(list(urm.row))
    items_list = np.asarray(list(urm.col))

    return users_list, items_list


# encode features in a list of numbers
def encode_features(feature_list_icm):
    le = preprocessing.LabelEncoder()
    le.fit(feature_list_icm)

    feature_list_icm = le.transform(feature_list_icm)
    print("encode features")
    print(feature_list_icm[0:10])
    return feature_list_icm


# create ICM indeces: (item, price_tag)
def get_icm_price():
    items_df = pd.read_csv("Data/data_ICM_price.csv") # read the csv into a dataframe
    items_list = items_df.row
    items_list = list(items_list)
    prices_list = items_df.data
    prices_list = list(prices_list)
    values_list = list(np.ones(len(prices_list)))

    ICM = sps.coo_matrix((values_list, (items_list, encode_features(prices_list))), dtype=np.float64)
    print(ICM.row)
    # ICM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(ICM)
    # ICM_tfidf = normalize(ICM_tfidf, axis=0, norm='l2')

    return ICM.tocsr()


# create ICM indeces: (item, asset_tag)
def get_icm_asset():
    items_df = pd.read_csv("Data/data_ICM_asset.csv")
    items_list = items_df.row
    items_list = list(items_list)
    assets_list = items_df.data
    assets_list = list(assets_list)
    values_list = list(np.ones(len(assets_list)))

    ICM = sps.coo_matrix((values_list, (items_list, encode_features(assets_list))), dtype=np.float64)
    # ICM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(ICM)
    # ICM_tfidf = normalize(ICM_tfidf, axis=0, norm='l2')

    return ICM.tocsr()


# create ICM indeces: (item, subclass)
def get_icm_sub_class():
    items_df = pd.read_csv("Data/data_ICM_sub_class.csv")
    items_list = items_df.row
    items_list = list(items_list)
    subclass_list = items_df.col
    subclass_list = list(subclass_list)
    values_list = list(np.ones(len(subclass_list)))

    ICM = sps.coo_matrix((values_list, (items_list, subclass_list)), dtype=np.float64)
    # ICM_tfidf = feature_extraction.text.TfidfTransformer().fit_transform(ICM)
    # ICM_tfidf = normalize(ICM_tfidf, axis=0, norm='l2')

    return ICM.tocsr()


def get_icm():
    items_data = pd.read_csv("Data/data_ICM_price.csv")
    prices = items_data.reindex(columns=['row', 'data'])
    prices.sort_values(by='row', inplace=True)  # this seems not useful, values are already ordered
    prices_list = [[a] for a in prices['data']]
    icm_prices = preprocessing.MultiLabelBinarizer(sparse_output=True).fit_transform(prices_list)  #create icm (item, price) and prices are encoded as in feature_encode function
    icm_prices_csr = icm_prices.tocsr()

    items_data = pd.read_csv("Data/data_ICM_asset.csv")
    assets = items_data.reindex(columns=['row', 'data'])
    assets.sort_values(by='row', inplace=True)  # this seems not useful, values are already ordered
    assets_list = [[a] for a in assets['data']]
    icm_assets = preprocessing.MultiLabelBinarizer(sparse_output=True).fit_transform(assets_list)
    icm_assets_csr = icm_assets.tocsr()

    items_data = pd.read_csv("Data/data_ICM_sub_class.csv")
    subclasses = items_data.reindex(columns=['row', 'col'])
    subclasses.sort_values(by='row', inplace=True)  # this seems not useful, values are already ordered
    subclasses_list = [[a] for a in subclasses['col']]
    icm_subclasses = preprocessing.MultiLabelBinarizer(sparse_output=True).fit_transform(subclasses_list)
    icm_subclasses_csr = icm_subclasses.tocsr()

    ICM = sps.hstack((icm_prices_csr, icm_assets_csr, icm_subclasses_csr))
    ICM_csr = ICM.tocsr()
    print("ciao" + ICM[:10, :])
    return ICM_csr


def create_test_set(urm_all):
    urm_all = urm_all.tocsr()
    shape = urm_all.shape
    urm_all_accessible = urm_all.tolil()
    np.random.seed(1234)

    urm_test = sps.lil_matrix(shape)

    for row in np.arange(urm_all.shape[0]):
        row_ptr = urm_all.indptr[row]
        row_ptr_end = urm_all.indptr[row + 1]

        # print(urm_all.indices[row_ptr: row_ptr_end])
        if len(urm_all.indices[row_ptr: row_ptr_end]) == 0:
            rand_index = 0
        else:
            rand_index = random.choice(urm_all.indices[row_ptr: row_ptr_end])
        # print(rand_index)
        # print(urm_all_accessible[row, rand_index])
        urm_all_accessible[row, rand_index] = 0
        urm_test[row, rand_index] = 1
        # print(urm_test[row, rand_index])
        # print(urm_all_accessible[row, rand_index])

    # print('stampaaaaaa')
    # print(urm_all_accessible[10, 10])
    # print(urm_test[10, 10])

    return urm_all_accessible.tocsr(), urm_test.tocsr()


def train_validation_holdout(urm_all, train_perc=0.8):
    num_interactions = urm_all.nnz
    urm_all = urm_all.tocoo()
    shape = urm_all.shape

    np.random.seed(1234)
    train_mask = np.random.choice([True, False], num_interactions, p=[train_perc, 1 - train_perc])

    urm_train = sps.coo_matrix((urm_all.data[train_mask], (urm_all.row[train_mask], urm_all.col[train_mask])),shape=shape)
    urm_train = urm_train.tocsr()

    valid_mask = np.logical_not(train_mask)

    urm_valid = sps.coo_matrix((urm_all.data[valid_mask], (urm_all.row[valid_mask], urm_all.col[valid_mask])), shape=shape)
    urm_valid = urm_valid.tocsr()

    return urm_train, urm_valid


def remove_cold_items(urm_all):
    warm_items_mask = np.ediff1d(urm_all.tocsc().indptr) > 0
    warm_items = np.arange(urm_all.shape[1])[warm_items_mask]

    urm_all = urm_all[:, warm_items]
    return urm_all


def remove_cold_users(urm_all):
    warm_users_mask = np.ediff1d(urm_all.tocsr().indptr) > 0
    warm_users = np.arange(urm_all.shape[0])[warm_users_mask]

    urm_all = urm_all[warm_users, :]
    return urm_all

