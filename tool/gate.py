import numpy as np

"""
가중치(w) = 입력 신호의 영향력
편향(b) = 뉴런의 민감도
"""
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])    #가중치
    b = -0.7                    #편향

    tmp = np.sum(x*w) + b

    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])    #가중치
    b = 0.7                    #편향

    tmp = np.sum(x*w) + b

    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])    #가중치
    b = -0.2                    #편향

    tmp = np.sum(x*w) + b

    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):

    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)

    return y