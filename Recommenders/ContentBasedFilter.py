from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np
from Base.utilFunctions import get_icm, get_icm_asset, get_icm_price, get_icm_sub_class

class ContentBasedFiltering(object):

    def __init__(self, urm, knn_alb=100, knn_art=300, shrink_alb=2, shrink_art=2, weight_alb=0.6):
        self.knn_album = knn_alb
        self.knn_artist = knn_art
        self.shrink_album = shrink_alb
        self.shrink_artist = shrink_art
        self.weight_album = weight_alb

        self.urm = urm

    def compute_similarity(self, ICM, knn, shrink):
        similarity_object = Compute_Similarity_Python(ICM.transpose(), shrink=knn, topK=shrink,
                                                     normalize=True, similarity="cosine")
        return similarity_object.compute_similarity()

    def fit(self):

        self.ICM_prices = get_icm_price()
        self.ICM_assets = get_icm_asset()
        self.ICM_subclasses = get_icm_sub_class()

        self.SM_prices = self.compute_similarity(self.ICM_prices, self.knn_album, self.shrink_album)
        self.SM_assets = self.compute_similarity(self.ICM_assets, self.knn_artist, self.shrink_artist)
        self.SM_subclasses = self.compute_similarity(self.ICM_subclasses, self.knn_artist, self.shrink_artist)

    def get_expected_ratings(self, user_id):
        user_id = int(user_id)
        liked_items = self.urm[user_id]
        expected_ratings_prices = liked_items.dot(self.SM_prices).toarray().ravel()
        expected_ratings_assets = liked_items.dot(self.SM_assets).toarray().ravel()
        expected_ratings_subclasses = liked_items.dot(self.SM_subclasses).toarray().ravel()


        expected_ratings = (expected_ratings_prices + expected_ratings_assets + expected_ratings_subclasses) / 3
        expected_ratings[liked_items.indices] = -np.inf
        return expected_ratings

    def recommend(self, user_id, at=10):
        user_id = int(user_id)
        expected_ratings = self.get_expected_ratings(user_id)

        recommended_items = np.flip(np.argsort(expected_ratings), 0)
        return recommended_items[:at]

### CODE TO TEST THIS ALG
# recommender = ContentBasedFiltering(knn_alb = 10,  knn_art = 10, shrink_alb = 0, shrink_art = 0, weight_alb = 0.7)
# Runner.run(is_test = True, recommender = recommender, split_type = None)
#########################