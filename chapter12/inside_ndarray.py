import pandas as pd
import numpy as np
from pandas import Series

def test():
    arr = np.ones((3, 4, 5))
    print(arr.strides) #跨度
#     类型
    ints = np.ones(10, dtype=np.uint16)

    floats = np.ones(10, dtype=np.float64)

    print(np.issubdtype(ints.dtype, np.integer))
    print(np.issubdtype(floats.dtype, np.floating))

    print(np.float64.mro()) #查看父类
    print(np.generic.mro()) #查看父类
    print(np.float.mro()) #查看父类

def operate():
    # reshape
    arr = np.arange(8)
    print(arr)
    arr2 = np.ones((2, 4))
    print(arr.reshape(arr2.shape))
    print(arr.reshape((4, 2)))
    print(arr.reshape((2, -1)))
    arr3 = arr.reshape(arr2.shape)
    print(arr3.ravel())
    print(arr3.flatten())
    # 行排序C 列排序
    print(arr3.ravel('F'))
    print(arr3.ravel('C'))
#     数组合并拆分
    print(np.concatenate([arr2, arr3], axis=0))
    print(np.concatenate([arr2, arr3], axis=1))
    print(np.vstack((arr2, arr3)))
    print(np.hstack((arr2, arr3)))

    arr4 = np.random.randn(5, 2)
    print(arr4)
    first, second, third = np.split(arr4, [1, 3])
    print(first)
    print(second)
    print(third)
    first, second, third = np.split(arr4, [2, 3])
    print(first)
    print(second)
    print(third)

#     r_ c_
    arr5= np.arange(4)
    print(np.r_[arr2, arr3])
    print(np.c_[np.r_[arr2, arr3], arr5])

    print(np.c_[1:6, -10:-5])

#     元素重复操作
    arr = np.arange(4)
    print(arr.repeat(3))
    print(arr.repeat([1, 2, 3, 4]))

    arr2 = arr.reshape((2, 2))
    print(arr2)
    print(arr2.repeat(2, axis=0))
    print(arr2.repeat([1, 2], axis=0))
    print(arr2.repeat(2, axis=1))
    print(arr2.repeat([1, 2], axis=1))

    print(np.tile(arr, 2))
    print(np.tile(arr, (3, 2)))

#     花式索引等价函数take put
    arr = np.arange(10) * 100
    print(arr)
    inds = [3, 1, 4]
    print(arr[inds])
    print(arr.take(inds))
    np.put(arr, inds, inds)
    print(arr)

    arr2 = arr.reshape((5,2))
    print(arr2)
    print(arr2.take([2, 1], axis=0))
    print(arr2[[2, 1]])


def broadcatiing():
    arr = np.arange(12).reshape((3, 4))
    print(arr)
    print(arr * 4)
    print(arr + 1)
    print(arr.mean(0))
    print(arr-arr.mean(0))
    print((arr-arr.mean(0)).mean(0))
    print(arr.mean(1))
    print(arr.mean(1))
    print((arr-arr.mean(1).reshape((3, 1))).mean(1))

    arr = np.zeros((4, 4))
    arr_3d = arr[:, np.newaxis, :]
    arr_3d2 = arr[np.newaxis, :, :]
    print(arr)
    print(arr_3d)
    print(arr_3d2)

    arr_1d= np.random.normal(size=3)
    print(arr_1d)
    print(arr_1d[:, np.newaxis])
    print(arr_1d[np.newaxis, :])

    arr = np.random.randn(3, 4, 5)
    print(arr.mean(2))
    print(arr.mean(1))
    print((arr-arr.mean(2)[:, :, np.newaxis]).mean(2))


def ufunc():
    arr = np.arange(10)
    print(np.add.reduce(arr))

    arr = np.arange(15).reshape((3, 5))
    print(arr)
    print(np.add.accumulate(arr, axis=1))

    arr = np.arange(3).repeat([1, 2, 2])
    print(arr)
    print(np.multiply.outer(arr, np.arange(5)))
    result = np.subtract.outer(np.random.randn(3, 4), np.random.randn(5))
    print(result)

    arr = np.arange(10)
    print(np.add.reduceat(arr, [0, 5, 8]))

    arr = np.multiply.outer(np.arange(4), np.arange(5))
    print(arr)
    print(np.add.reduceat(arr, [0, 2, 4], axis=1))

    add_them = np.frompyfunc(add_elements, 2, 1)
    print(add_them(np.arange(8), np.arange(8)))

    add_them = np.vectorize(add_elements, otypes=[np.int])
    print(add_them(np.arange(8), np.arange(8)))


def add_elements(x, y):
    return x + y


def struct_record():
    dtype = [('x', np.float64), ('y', np.int32)]
    arr = np.array([(1.5, 4), (3.2, 4)], dtype=dtype)
    print(arr.dtype)
    print(arr)


def sort():
    arr = np.random.randn(5)
    print(arr)
    arr.sort()
    print(arr)

    arr = np.random.randn(3, 5)
    print(arr)
    arr[:, 0].sort()
    print(arr)

    arr = np.random.randn(5)
    print(arr)
    print(np.sort(arr))
    print(arr)

    arr = np.random.randn(3, 5)
    print(arr)
    arr.sort(axis=1)
    print(arr)


def argsort_lexsort():
    values = np.array([4, 2, 5, 1, 7])
    indexer = values.argsort()
    # print(indexer)
    # print(values[indexer])

    first_name = np.array(['Bob', 'Jane', 'Steve', 'Bill', 'Barbara'])
    last_name = np.array(['Jones', 'Arnold', 'Arnold', 'Jones', 'Walters'])
    sorter = np.lexsort((first_name, last_name))
    print('--------------------')
    print(sorter)
    x = zip(last_name[sorter], first_name[sorter])
    for x, y in x:
        print(x + ':' + y)


def searchsorted():
    arr = np.array([0, 1, 7, 12, 15])
    print(arr.searchsorted(9))

    print(arr.searchsorted([2, 1, 23]))
    print(arr.searchsorted([2, 1, 23], side='right'))

    data = np.floor(np.random.uniform(0, 10000, size=50))
    bins = np.array([0, 100, 1000, 5000, 10000])
    print(data)
    labels = bins.searchsorted(data)
    print(labels)
    print(Series(data).groupby(labels).mean())

    print(np.digitize(data, bins))

def matrix():
    X = np.random.randn(4, 4)
    xm = np.matrix(X)
    ym = xm[:, 0]
    print(xm)
    print(ym)
    print(ym.T * xm * ym)

    print(xm.I * X)


def memmap():
    mmap = np.memmap('mymmap', dtype=np.float64, mode='w+', shape=(10000, 10000))
    print(mmap)
    section = mmap[:5]
    section[:] = np.random.randn(5, 10000)
    mmap.flush()
    print(mmap)
    mmap = np.memmap('mymmap', dtype=np.float64, shape=(10000, 10000))
    print(mmap)

if __name__ == '__main__':
    # test()
    # operate()
    # a = np.arange(5)
    # c = np.put(a, [0, 2], [-44, -55])
    # print(a)
    # broadcatiing()
    # ufunc()
    # struct_record()
    # sort()
    # argsort_lexsort()
    # searchsorted()
    # matrix()
    memmap()