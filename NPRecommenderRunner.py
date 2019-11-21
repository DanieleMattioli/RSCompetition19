import scipy.sparse as sps
import numpy as np
import pandas as pd
from tqdm import tqdm

from Base import utilFunctions as f, evaluation_function as evaluation
from Recommenders.TopPop import TopPopRecommender
from Recommenders.GlobalEffects import GlobalEffects

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

# split into train and test URM
urm_train, urm_test = f.train_validation_holdout(urm_all)

# train the recommender: TopPopular
# -----------------------------------------
recommender = TopPopRecommender()
recommender.fit(urm_train)

# recommend 10 items to 10 random users among those in the icm
userList_unique = list(set(userList))
for user_id in userList_unique[0:10]:
    print(recommender.recommend(user_id, at=10))

# -----------------------------------------

# evaluate the recommender
evaluation.evaluate_algorithm(urm_test, recommender, 10)
# TopPop performance is: Precision = 0.0079, Recall = 0.0232, MAP = 0.0072


# # train the recommender: GlobalEffects
# # -----------------------------------------
# recommender = GlobalEffects()
# recommender.fit(urm_train)
#
# # recommend 10 items to 10 random users among those in the icm
# userList_unique = list(set(userList))
# for user_id in userList_unique[0:10]:
#     print(recommender.recommend(user_id, at=10))
#
# # -----------------------------------------
#
# # evaluate the recommender
# evaluation.evaluate_algorithm(urm_test, recommender, 10)
# # GE performance is: Precision = 0.0005, Recall = 0.0011, MAP = 0.0003


# -----------------------------------------
# submission
# -----------------------------------------

submission_file = open("Data/submission.csv", "w")
submission_file.write("user_id,item_list\n")

sample_file = open('Data/data_target_users_test.csv', 'r')
sample_file = pd.read_csv(sample_file)
users_to_test = np.asarray(list(sample_file.user_id))
print(len(users_to_test))

for user_id in tqdm(users_to_test):
    result = recommender.recommend(user_id)
    result = " ".join(str(x) for x in result)
    submission_file.write(str(user_id) + "," + str(result) + "\n")

submission_file.close()
print("Saved predictions to file")



