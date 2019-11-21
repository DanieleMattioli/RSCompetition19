import scipy.sparse as sps
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
import pandas as pd

from Base import utilFunctions as f, evaluation_function as evaluation
from Recommenders.IBCollaborativeFilter import ItemBasedCollaborativeFilter
from Recommenders.ContentBasedFilter import ContentBasedFiltering
from Recommenders.UBCollaborativeFilter import UserBasedCollaborativeFilter
# from Recommenders.SLIMBPR import SLIM_BPR
from Recommenders.OldSLIM import SLIM_BPR
#from HyperparametersSearch.IBCparameters import IBCparamsearch

# load data
urm_path = "Data/data_train.csv"
urm_file = open(urm_path, 'r')

# read URM file and produce lists of items and users
userList, itemList = f.get_urm_lists(urm_file)

# not unique lists of users, items and ratings
userList = list(userList)
itemList = list(itemList)
ratingList = list(np.ones(len(userList)))

# URM matrix (explicit ratings)
urm_all = sps.coo_matrix((ratingList, (userList, itemList)))
urm_all = urm_all.tocsr()

# separate test set from urm_all: from now on, urm_all = urm_train and urm_valid, while urm_test is isolated
# when creating submission file: no need of splitting data
urm_all, urm_test = f.create_test_set(urm_all)

print('done test')

#
# # -----------------------------------------------------------------------------------
# # the following evaluation uses a k-fold cross validation sets to evaluate the recommenders
# # -----------------------------------------------------------------------------------
#
# k_splits = 10
# kf = KFold(n_splits=k_splits)
#
# tot_performance = 0
# k = 0
#
# print('evaluating ItemBasedCollaborative')
#
# for train_index, test_index in kf.split(urm_all):
#
#     k += 1
#
#     # at each iteration we create a different train and validation set
#     urm_train, urm_valid = urm_all[train_index], urm_all[test_index]
#
#     # we train the recommender on the train_urm and evaluate it on the valid_urm
#     recommender = ItemBasedCollaborativeFilter(urm_train)
#     recommender.fit()
#
#     performance = evaluation.evaluate_algorithm(urm_valid, recommender, 10)['MAP']
#     print("performance in train test " + str(k) + " is: " + str(performance))
#
#     # we collect the performace to average it at the end
#     tot_performance += performance
#
# tot_performance /= k_splits
# print('the performance out of k-fold crossvalidation is ' + str(tot_performance))
# # ItemBasedCollaborativeFilter: k-fold crossvalidation performance is 0.004961225925902897
#
# # -----------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------
# the following evaluation uses a single validation set to evaluate the recommenders
# -----------------------------------------------------------------------------------

# split into train and validation URM: use for tuning parameters
urm_train, urm_valid = f.train_validation_holdout(urm_all)

# # run only to tune parameters, then choose the correct ones and comment code
# paramsearch = IBCparamsearch(urm_train, urm_valid)
# paramsearch.runsearch()

# # choosen topK and shrink
# topK = 500
# shrink = 10
#
# # ---------------------------------------------------------------------------------
# # train the recommender:  ItemBasedCollaborative
# # ---------------------------------------------------------------------------------
# IBC_recommender = ItemBasedCollaborativeFilter(urm_all)
# IBC_recommender.fit(shrink=shrink, topK=topK)
#
# # evaluate the recommender
# print("item based collaborative filter: ")
# # evaluation.evaluate_algorithm(urm_test, IBC_recommender, 10)
# # Recommender performance is: Precision = 0.0265, Recall = 0.0793, MAP = 0.0436
# # ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# train the recommender:  ContentBased
# ---------------------------------------------------------------------------------
#CBF_recommender = ContentBasedFiltering(urm_train)
#CBF_recommender.fit()

# evaluate the recommender
#print("content based filter: ")
#evaluation.evaluate_algorithm(urm_valid, CBF_recommender, 10)
# Recommender performance is: Precision = 0.0021, Recall = 0.0071, MAP = 0.0029
# ---------------------------------------------------------------------------------

#
# # ---------------------------------------------------------------------------------
# # train the recommender:  UserBasedCollaborative
# # ---------------------------------------------------------------------------------
# UBC_recommender = UserBasedCollaborativeFilter(urm_train)
# UBC_recommender.fit()
#
# # evaluate the recommender
# print("user based collaborative filter: ")
# evaluation.evaluate_algorithm(urm_valid, UBC_recommender, 10)
# # Recommender performance is: Precision = 0.0112, Recall = 0.0311, MAP = 0.0154
# # ---------------------------------------------------------------------------------
#
#
#
# # ---------------------------------------------------------------------------------
# # train the recommender:  SLIM BPR
# # ---------------------------------------------------------------------------------
SLIMBPR_recommender = SLIM_BPR(urm_train)
#SLIMBPR_recommender.fit()
SLIMBPR_recommender.fit(learning_rate=0.04865518400065951, epochs=50, k=10)
#
# # evaluate the recommender
print("SLIM BPR: ")
evaluation.evaluate_algorithm(urm_valid, SLIMBPR_recommender, 10)
# # Recommender performance is: Precision = 0.0112, Recall = 0.0311, MAP = 0.0154
# # ---------------------------------------------------------------------------------
#

# ---------------------------------------------------------------------------------
# create the submission file
# ---------------------------------------------------------------------------------

# choose the recommender to submit
recommender = SLIMBPR_recommender

# write predictions in the output file
submission_file = open("Data/submission.csv", "w")
submission_file.write("user_id,item_list\n")

sample_file = open('Data/data_target_users_test.csv', 'r')
sample_file = pd.read_csv(sample_file)
users_to_test = np.asarray(list(sample_file.user_id))

print(len(users_to_test))

# for user_id in tqdm(set(userList)):   # this would iterate only on users with interactions (specified in the urm file)
# for user_id in tqdm(np.arange(0, max(set(userList)))):  # this iterates over all users (even with no interactions)
for user_id in tqdm(users_to_test):
    result = recommender.recommend(user_id)
    result = " ".join(str(x) for x in result)
    submission_file.write(str(user_id) + "," + str(result) + "\n")

submission_file.close()
print("Saved predictions to file")
