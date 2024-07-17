from numpy import linspace


def KarnikMendel(it2fs):
    N = 10001
    x = linspace(0, 10, N)
    theta = [(it2fs.lower_mf(i) + it2fs.upper_mf(i)) / 2 for i in x]
    ccl = sum(x * theta) / sum(theta)
    theta = [it2fs.upper_mf(i) if i <= ccl else it2fs.lower_mf(i) for i in x]
    ckl = sum(x * theta) / sum(theta)
    while abs(ccl - ckl) > 1e-5:
        ccl = ckl
        theta = [it2fs.upper_mf(i) if i <=
                 ccl else it2fs.lower_mf(i) for i in x]
        ckl = sum(x * theta) / sum(theta)

    theta = [(it2fs.lower_mf(i) + it2fs.upper_mf(i)) / 2 for i in x]
    ccr = sum(x * theta) / sum(theta)
    theta = [it2fs.lower_mf(i) if i <= ccr else it2fs.upper_mf(i) for i in x]
    ckr = sum(x * theta) / sum(theta)
    while abs(ccr - ckr) > 1e-5:
        ccr = ckr
        theta = [it2fs.lower_mf(i) if i <=
                 ccr else it2fs.upper_mf(i) for i in x]
        ckr = sum(x * theta) / sum(theta)
    return ckl, ckr
