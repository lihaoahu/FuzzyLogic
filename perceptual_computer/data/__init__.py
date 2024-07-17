from pandas import DataFrame,read_excel,read_csv
from perceptual_computer import BASE_DIR

def csv_loader(csv):
    # 读取数据
    collected_data_intervals = read_csv(csv)
    # 获取区间数据对应语言术语集
    words = collected_data_intervals.columns[0:-1:2]

    # 以语言术语为键，对应区间数据为值，将数据存储为字典
    intervals = dict()
    for i, word in enumerate(words):
        l, r = collected_data_intervals.iloc[:, 2 * i].tolist(), collected_data_intervals.iloc[:,
                                                                          2 * i + 1].tolist()
        intervals[word] = DataFrame({"left": l, "right": r})
    return intervals



# 读取数据
collected_data_intervals_175subjects = read_excel(io = BASE_DIR / "data/175subjects.xls")
# 获取区间数据对应语言术语集
words_175subjects = collected_data_intervals_175subjects.columns[0:-1:2]

# 以语言术语为键，对应区间数据为值，将数据存储为字典
intervals_175subjects = dict()
for i, word in enumerate(words_175subjects):
    l, r = collected_data_intervals_175subjects.iloc[:, 2*i].tolist(), collected_data_intervals_175subjects.iloc[:, 2*i+1].tolist()
    intervals_175subjects[word] = DataFrame({"left": l, "right": r})


# 读取数据
collected_data_intervals_28subjects = read_excel(io = BASE_DIR / "data/28subjects.xls")
# 获取区间数据对应语言术语集
words_28subjects = collected_data_intervals_28subjects.columns[0:-1:2]

# 以语言术语为键，对应区间数据为值，将数据存储为字典
intervals_28subjects = dict()
for i, word in enumerate(words_28subjects):
    l, r = collected_data_intervals_28subjects.iloc[:, 2*i].tolist(), collected_data_intervals_28subjects.iloc[:, 2*i+1].tolist()
    intervals_28subjects[word] = DataFrame({"left": l, "right": r})

intervals_dianping = csv_loader(BASE_DIR / "data/dianping.csv")


__all__ = [
    "intervals_28subjects","intervals_175subjects","intervals_dianping","csv_loader"
]
