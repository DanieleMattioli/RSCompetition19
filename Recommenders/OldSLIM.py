#   epochs  |   k   |   MAP on valid    |
#     15        100         0.0224
#     15        50          0.0219
#     15        200         0.0217
#     30        100         0.0258
#     50        100         0.0273
import numpy as np
import time
from Base.Recommender_utils import similarityMatrixTopK


class SLIM_BPR(object):
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def __init__(self, URM):
        self.URM = URM
        self.URM_mask = self.URM.copy()
        self.URM_mask.eliminate_zeros()

        self.n_users = self.URM_mask.shape[0]
        self.n_items = self.URM_mask.shape[1]

        # Extract users having at least one interaction to choose from
        self.eligible_users = []

        self.similarity_matrix = np.zeros((self.n_items, self.n_items))

        for user_id in range(self.n_users):

            start_pos = self.URM_mask.indptr[user_id]
            end_pos = self.URM_mask.indptr[user_id + 1]

            if len(self.URM_mask.indices[start_pos:end_pos]) > 0:
                self.eligible_users.append(user_id)

    def sampleTriplet(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)

        user_id = np.random.choice(self.eligible_users)

        # Get user seen items and choose one
        user_liked_items = self.URM_mask[user_id, :].indices
        pos_item_id = np.random.choice(user_liked_items)

        neg_item_selected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not neg_item_selected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in user_liked_items):
                neg_item_selected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self):

        # Get number of available interactions
        num_positive_iteractions = int(self.URM_mask.nnz * 0.01)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        # Uniform user sampling without replacement
        for num_sample in range(num_positive_iteractions):

            # Sample
            user_id, positive_item_id, negative_item_id = self.sampleTriplet()

            user_liked_items = self.URM_mask[user_id, :].indices

            # Prediction
            x_i = self.similarity_matrix[positive_item_id, user_liked_items].sum()
            x_j = self.similarity_matrix[negative_item_id, user_liked_items].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            # Update
            self.similarity_matrix[positive_item_id, user_liked_items] += self.learning_rate * gradient
            self.similarity_matrix[positive_item_id, positive_item_id] = 0

            self.similarity_matrix[negative_item_id, user_liked_items] -= self.learning_rate * gradient
            self.similarity_matrix[negative_item_id, negative_item_id] = 0

            if (time.time() - start_time_batch >= 30 or num_sample == num_positive_iteractions - 1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample) / num_positive_iteractions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()

    def fit(self, learning_rate=0.01, epochs=50, k=100):

        self.learning_rate = learning_rate
        self.epochs = epochs

        for numEpoch in range(self.epochs):
            self.epochIteration()

        self.similarity_matrix = self.similarity_matrix.T

        self.similarity_matrix = similarityMatrixTopK(self.similarity_matrix, k=k)

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.similarity_matrix).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores