from Recommenders.IBCollaborativeFilter import ItemBasedCollaborativeFilter
from Base import evaluation_function as evaluation
import matplotlib.pyplot as pyplot


class IBCparamsearch():
    # hyperparameters search

    def __init__(self, urm_train, urm_valid):
        self.urm_train = urm_train
        self.urm_valid = urm_valid

    def runsearch(self):
        # -----------------------------------------------------------------------------------
        x_tick = [500, 700, 1000]
        MAP_per_k = []

        for topK in x_tick:
            recommender = ItemBasedCollaborativeFilter(self.urm_train)
            recommender.fit(shrink=0.0, topK=topK)

            result_dict = evaluation.evaluate_algorithm(self.urm_valid, recommender)
            MAP_per_k.append(result_dict["MAP"])

        pyplot.plot(x_tick, MAP_per_k)
        pyplot.ylabel('MAP')
        pyplot.xlabel('TopK')
        pyplot.show()

        # shrink = 0, topK = {500, 700, 1000}
        # { MAP = 0.0214, MAP = 0.0223, MAP = 0.0239}   here keeps increasing
        # max = 0.02389758906468615 (topK=1000)
        print(max(MAP_per_k))
        # -----------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------
        x_tick = [500, 700, 1000]
        MAP_per_k = []

        for topK in x_tick:
            recommender = ItemBasedCollaborativeFilter(self.urm_train)
            recommender.fit(shrink=10.0, topK=topK)

            result_dict = evaluation.evaluate_algorithm(self.urm_valid, recommender)
            MAP_per_k.append(result_dict["MAP"])

        pyplot.plot(x_tick, MAP_per_k)
        pyplot.ylabel('MAP')
        pyplot.xlabel('TopK')
        pyplot.show()

        # shrink = 10, topK = {500, 700, 1000}
        # { MAP = 0.0375, MAP = 0.0375, MAP = 0.0379}   still increasing
        # max = 0.037863194967651136 (topK=1000)
        print(max(MAP_per_k))
        # -----------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------
        x_tick = [500, 700, 1000]
        MAP_per_k = []

        for topK in x_tick:
            recommender = ItemBasedCollaborativeFilter(self.urm_train)
            recommender.fit(shrink=50.0, topK=topK)

            result_dict = evaluation.evaluate_algorithm(self.urm_valid, recommender)
            MAP_per_k.append(result_dict["MAP"])

        pyplot.plot(x_tick, MAP_per_k)
        pyplot.ylabel('MAP')
        pyplot.xlabel('TopK')
        pyplot.show()

        # shrink = 50, topK = {500, 700, 1000}
        # { MAP = 0.0371, MAP = 0.0368, MAP = 0.0362}   it is decreasing (maybe lower topK and same shrink is good)
        # max = 0.03708426015618488 (topK=500)
        print(max(MAP_per_k))
        # -----------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------
        x_tick = [500, 700, 1000]
        MAP_per_k = []

        for topK in x_tick:
            recommender = ItemBasedCollaborativeFilter(self.urm_train)
            recommender.fit(shrink=100.0, topK=topK)

            result_dict = evaluation.evaluate_algorithm(self.urm_valid, recommender)
            MAP_per_k.append(result_dict["MAP"])

        pyplot.plot(x_tick, MAP_per_k)
        pyplot.ylabel('MAP')
        pyplot.xlabel('TopK')
        pyplot.show()

        print(max(MAP_per_k))
        # shrink = 50, topK = {500, 700, 1000}
        # { MAP = 0.0352, MAP = , MAP =  }   decreasing too much: stop! (maybe lower topK and same shrink is good)
        # max =