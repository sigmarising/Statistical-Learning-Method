import numpy as np

#np.random.seed(3)

def init(pi, p, q, booksdata=False, N = 200):

    if booksdata:
        Y = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    else:
        print("产生数据的参数为 pi:%.2f, p:%.2f, q:%.2f" % (pi, p, q))
        A = np.random.binomial(1, pi, [N])
        print("硬币A的序列Z", A)

        Y = []

        for a in A:
            if a == 1:
                y = np.random.binomial(1, p)
            else:
                y = np.random.binomial(1, q)

            Y.append(y)
        print("\n观测序列Y", Y)
        print("观测序列共%d个1，%d个0\n" % (Y.count(1), Y.count(0)))
    return Y


def Tri_coin_EM_algorithm(Y, pi, p ,q):
    #step1: inition
    n = 0
    breakflag = 0

    while True:
        print("第", n, "次迭代：", "pi:", pi, "p:", p, "q:", q)

        # step2: Expection

        mu = []
        for y in Y:
            muj = (pi * (p ** y) * ((1 - p) ** (1 - y))) / (
                        pi * (p ** y) * ((1 - p) ** (1 - y)) + (1 - pi) * q ** y * ((1 - q) ** (1 - y)))
            mu.append(muj)
        #print(mu)

        # step3: Maximum

        summu = sum(mu)
        sumy = sum(Y)
        summuy = 0
        for i in range(len(Y)):
            summuy = summuy + mu[i] * Y[i]

        #print(summu, sumy, summuy)

        npi = summu / len(Y)
        np = summuy / summu
        nq = (sumy - summuy) / (len(Y) - summu)

        #step4: converge

        if abs(npi-pi) < 1e-10 and abs(np-p) < 1e-10 and abs(nq-q) < 1e-10:
            breakflag += 1
            if breakflag >1:
                print("最后一次迭代:", "pi:", pi, "p:", p, "q:", q)
                break
            else:
                pi = npi
                p = np
                q = nq
                n = n + 1
        else:
            pi = npi
            p = np
            q = nq
            n = n+1


if __name__ == '__main__':
    Y = init(pi=0.4, p=0.8, q=0.4, booksdata=False)
    Tri_coin_EM_algorithm(Y, pi=0.4, p=0.1, q=0.8)
