import numpy as np
from  numpy.random import randn
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import sys
from numpy.linalg import eigvals
def init():
    data = {i:randn() for i in range(7)}
    print data

def app():
    app=QApplication(sys.argv)
    b=QPushButton("Hello Kitty!")
    b.show()
    app.connect(b,SIGNAL("clicked()"),app,SLOT("quit()"))
    app.exec_()

def run_experiment(nither = 100):
    k =100
    results = []
    for _ in xrange(nither):
        mat = np.random.rand(k,k)
        max_eigenvalue = np.abs(eigvals(mat)).max()
        results.append(max_eigenvalue)
    return results
if __name__=='__main__':
    some_results = run_experiment()
    print 'Largest one we saw : %s' % np.max(some_results)

