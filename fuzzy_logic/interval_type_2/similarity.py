from numpy import linspace


def jaccard_similarity(it2fs_1, it2fs_2):
    x = linspace(0, 10, 1001)
    it2fs_1_lmf = it2fs_1.lower_mf(x)
    it2fs_1_umf = it2fs_1.upper_mf(x)
    it2fs_2_lmf = it2fs_2.lower_mf(x)
    it2fs_2_umf = it2fs_2.upper_mf(x)
    return (sum(min([it2fs_1_lmf, it2fs_2_lmf], axis=0)) + sum(min([it2fs_1_umf, it2fs_2_umf], axis=0))) / (
        sum(max([it2fs_1_lmf, it2fs_2_lmf], axis=0)) + sum(max([it2fs_1_umf, it2fs_2_umf], axis=0)))
