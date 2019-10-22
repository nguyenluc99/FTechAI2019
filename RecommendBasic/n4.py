import numpy as np
if __name__ != '__main__':
    # n = int(input())
    # marksheet = [[input(), float(input())] for _ in range(n)]
    # second_highest = sorted(list(set([marks for name, marks in marksheet])))[1]
    # print('\n'.join([a for a,b in sorted(marksheet) if b == second_highest]))

    # lst = [1, 2]
    # print(lst.index(3))
    # earned = 0
    # len = int(input())
    # lst_shoe = input().split(' ')
    # for ind in lst_shoe : 
    #     ind = int(ind)
    # for _ in range(int(input())) :
        

    # print(earned)
    # lst = ['hi', 'this is a string']
    # s = 0
    # s += a for a in lst
    # print(''.join(str(a) for a in range(1, int(input()) + 1)))
    l1 = [1,2,3, 4]
    l2 = ['one', 'two', 'three','four', 'five']
    for x,y in zip(l1, l2) :
        print(x, ' is ' , y)
    import numpy as np
    import matplotlib.pyplot as plt
    # %matplotlib inline
    def cost(theta_0, theta_1, xs, ys):
        distance = 0
        for x,y in zip(xs, ys):
            predicted_value = theta_0 + theta_1 * x
            distance = distance + abs(y - predicted_value)
        return distance/len(xs)
    areas = [1000, 2000, 4000]
    prices = [200000, 250000, 300000]
    theta_0 = 0
    testno = 200
    costs = [cost(theta_0, theta_1, areas, prices) for theta_1 in np.arange(testno)]
    plt.plot(np.arange(testno), costs)
    print(np.arange(testno), costs)
    plt.show()

lst = np.zeros_like(10)
print((lst))


