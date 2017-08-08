import random
from collections import Counter
import matplotlib.pyplot as plt


def random_sample(sequence):
    """
    根据经验CDF来抽样
    键值元组构成序列参数seq[(key1, value1), (key2, value2), ...]
    """
    # 键由小到大排序，实际seq可能是其他离散分布
    seq_new = sorted(sequence, key=lambda pair: pair[0])
    # 获得一个均匀分布的随机数
    u = random.random()
    for k, v in seq_new:
        # 当前值小于随机数u，不抽出样本
        if v < u:
            # CDF，累积概率
            u -= v
            continue
        # 抽出样本
        else:
            return k


if __name__ == '__main__':
    seq = [(2, 0.2), (1, 0.3), (3, 0.5)]

    # 抽样1000次
    res = []
    for i in range(1000):
        res.append(random_sample(seq))

    res_dict = Counter(res)
    plt.xticks([0, 1, 2, 3])
    plt.bar(list(res_dict.keys()), list(res_dict.values()))
    for x, y in zip(list(res_dict.keys()), list(res_dict.values())):
        plt.text(x, y + 20, '%.2f' % (y / 1000), ha='center', va='top')
    plt.show()

