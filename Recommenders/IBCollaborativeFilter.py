from Base.Similarity.Compute_Similarity_Python import Compute_Similarity_Python
import numpy as np


class ItemBasedCollaborativeFilter(object):

    def __init__(self, urm):
        self.urm = urm

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object = Compute_Similarity_Python(self.urm, shrink=shrink, topK=topK,
                                                      normalize=normalize, similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()

    def recommend(self, user_id, at=10, exclude_liked=True):
        # compute the scores using the dot product
        user_profile = self.urm[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_liked:
            scores = self.filter_liked(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_liked(self, user_id, scores):
        start_pos = self.urm.indptr[user_id]
        end_pos = self.urm.indptr[user_id + 1]

        user_profile = self.urm.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

