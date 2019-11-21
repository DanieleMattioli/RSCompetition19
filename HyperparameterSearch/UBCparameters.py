from Recommenders.UBCollaborativeFilter import UserBasedCollaborativeFilter
from Base import evaluation_function as evaluation
import matplotlib.pyplot as pyplot


class IBCparamsearch():
    # hyperparameters search

    def __init__(self, urm_train, urm_valid):
        self.urm_train = urm_train
        self.urm_valid = urm_valid

    def runsearch(self):
        # -----------------------------------------------------------------------------------
        x_tick = [10, 50, 100, 200, 500, 700, 1000]
        MAP_per_k = []

        for topK in x_tick:
            recommender = UserBasedCollaborativeFilter(self.urm_train)
            recommender.fit(shrink=0.0, topK=topK)

            result_dict = evaluation.evaluate_algorithm(self.urm_valid, recommender)
            MAP_per_k.append(result_dict["MAP"])

        pyplot.plot(x_tick, MAP_per_k)
        pyplot.ylabel('MAP')
        pyplot.xlabel('TopK')
        pyplot.show()

        # shrink = 0, topK = {10, 50, 100, 200, 500, 700, 1000}
        #
        # max =
        print(max(MAP_per_k))
        # -----------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------
        x_tick = [500, 700, 1000]
        MAP_per_k = []

        for topK in x_tick:
            recommender = UserBasedCollaborativeFilter(self.urm_train)
            recommender.fit(shrink=10.0, topK=topK)

            result_dict = evaluation.evaluate_algorithm(self.urm_valid, recommender)
            MAP_per_k.append(result_dict["MAP"])

        pyplot.plot(x_tick, MAP_per_k)
        pyplot.ylabel('MAP')
        pyplot.xlabel('TopK')
        pyplot.show()

        # shrink = 10, topK = {500, 700, 1000}
        #
        # max =
        print(max(MAP_per_k))
        # -----------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------
        x_tick = [10, 50, 100, 200, 500]
        MAP_per_k = []

        for topK in x_tick:
            recommender = UserBasedCollaborativeFilter(self.urm_train)
            recommender.fit(shrink=50.0, topK=topK)

            result_dict = evaluation.evaluate_algorithm(self.urm_valid, recommender)
            MAP_per_k.append(result_dict["MAP"])

        pyplot.plot(x_tick, MAP_per_k)
        pyplot.ylabel('MAP')
        pyplot.xlabel('TopK')
        pyplot.show()

        # shrink = 50, topK = {10, 50, 100, 200, 500}
        # {
        # max =
        print(max(MAP_per_k))
        # -----------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------
        x_tick = [10, 50, 100, 200]
        MAP_per_k = []

        for topK in x_tick:
            recommender = UserBasedCollaborativeFilter(self.urm_train)
            recommender.fit(shrink=100.0, topK=topK)

            result_dict = evaluation.evaluate_algorithm(self.urm_valid, recommender)
            MAP_per_k.append(result_dict["MAP"])

        pyplot.plot(x_tick, MAP_per_k)
        pyplot.ylabel('MAP')
        pyplot.xlabel('TopK')
        pyplot.show()

        print(max(MAP_per_k))
        # shrink = 50, topK = {10, 50, 100, 200}
        # {
        # max =