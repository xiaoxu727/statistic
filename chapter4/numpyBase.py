import numpy as np
import matplotlib.pyplot as plt
import random


def main():
    data =[[5,6,7,8],[1,2,3,4]]
    data = np.array(data)
    print data
    print data*10
    print data+data
    print data.shape
    print data.dtype
    print data.ndim
    print data.T
    data.sort()
    print data
def imgshow():
    points = np.arange(-5,5,0.01)
    xs,ys = np.meshgrid(points,points)
    z = np.sqrt(xs**2+ys**2)
    plt.imshow(z, cmap=plt.cm.gray)
    plt.colorbar()
    plt.title('Image plot')
    print z
def random_test():
    position = 0
    walk = [position]
    steps = 1000
    for i in xrange(steps):
        step = 1 if random.randint(0, 1) else -1
        position += step
        walk.append(position)
    print walk
if __name__ == '__main__':
    # random_test()
    main()