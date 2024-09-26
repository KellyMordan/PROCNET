class UtilStructure:
    def __init__(self):
        pass

    @staticmethod
    def find_max_number_index(x: list):
        res = x.index(max(x))
        return res #寻找列表x中的最大值。返回最大值在列表x中的索引位置。

    @staticmethod
    def find_max_and_number_index(x: list):
        max_number = max(x)
        index = x.index(max_number)
        return max_number, index
