import numpy as np


class FrequentItemsetMining:
    def __init__(self):
        pass

    def generate_random_dataset(self):
        T = int(input("Enter the number of transactions: "))
        I = int(input("Enter the number of distinct items: "))
        data = dict()
        for i in T:
            random_size = np.random.randint(low=1, high=I + 1, size=1)[0]
            data[i + 1] = np.random.randint(low=1, high=I + 1,
                                            size=random_size)

        return data

    # def apriori(self, itemset:set, data: dict, min_support = 2):
    #     freq_itemsets = dict()
    #     itemset_size = 1
    #     temp_itemset = dict()
    #     while True:
    #         for item in itemset:
    #             for value in data.values():
    #                 if ite
