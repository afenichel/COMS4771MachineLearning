import scipy
from scipy import io
from scipy import ndimage
from scipy import stats
import numpy as np
import numpy.matlib
import math

#m is the training size
#c is the class - 0,1,2,3,4,5,6,7,8,9
#k is the index of the test data

#part i. classifier per class c=[0,1,2,3,4,5,6,7,8,9]
def hw_v0(c, T, training_x, training_y):
	training_ybin=training_y.astype(int)	
	training_ybin[training_ybin!=c]=np.negative(1)
	training_ybin[training_ybin==c]=1
	d=np.size(training_x,1)
	n=np.size(training_x,0)

	w={}
	w[0]=np.zeros(d)
	for t in range(T):
		i=(t+1)%(n+1)
		if training_ybin[i-1]*np.dot(w[t], training_x[i-1,:]) <= 0:
			w[t+1]=w[t]+training_ybin[i-1]*training_x[i-1,:]
		else: 
			w[t+1]=w[t]

	return w[T]


#part i. test for one input x_test[k] on all classes c
def classifier_v0(m,k,T):
	mat = scipy.io.loadmat('hw1data.mat')
	X=mat['X']
	Y=mat['Y']

	x_std=stats.zscore(X, axis=0)
	mu=np.mean(x_std, axis=0)
	sd=np.std(X, axis=0)
	x_std=x_std[:, ~np.isnan(mu)]

	d=np.size(x_std,1)
	n=np.size(x_std,0)

	indices = np.random.permutation(n)

	training_idx, test_idx = indices[:m], indices[m:]
	
	test_x=x_std[test_idx, :]
	test_y =Y[test_idx,:]

	training_x=x_std[training_idx,:]
	training_y=Y[training_idx,:]
	
	y_list=[i for i in range(10)]
	y_list=np.random.permutation(y_list)
	g={}
	f={}
	test_input=test_x[k,:]
	for c in y_list:
		wT=hw_v0(c, T, training_x, training_y)
		g[c]=np.dot(wT, test_input)
		f[c]=np.sign(np.dot(wT, test_input))

	print 'Perceptron V0 output:'
	print 'guess=',  max(g, key=g.get)
	print 'y_actual=',test_y[k]

	return max(g, key=g.get), test_y[k]
	


#part ii. loop through all testing inputs
def test_v0(m, T, training_x, training_y, test_x, test_y):
	y_list=[i for i in range(10)]
	y_list=np.random.permutation(y_list)
	w={}
	for c in y_list:
		w[c]=hw_v0(c, T, training_x, training_y)
	return w



def accuracy_v0(m, T):
	mat = scipy.io.loadmat('hw1data.mat')
	X=mat['X']
	Y=mat['Y']

	x_std=stats.zscore(X, axis=0)
	mu=np.mean(x_std, axis=0)
	sd=np.std(X, axis=0)
	x_std=x_std[:, ~np.isnan(mu)]

	d=np.size(x_std,1)
	n=np.size(x_std,0)

	indices = np.random.permutation(n)
	training_idx, test_idx = indices[:m], indices[m:]
	
	test_x=x_std[test_idx, :]
	test_y =Y[test_idx,:]

	training_x=x_std[training_idx,:]
	training_y=Y[training_idx,:]

	w=test_v0(m, T, training_x, training_y, test_x, test_y)
	accuracy=[]
	y_list=[i for i in range(10)]
	y_list=np.random.permutation(y_list)


	for k in range(1000):
		f={}
		g={}
		test_input=test_x[k,:]
		for c in y_list:
			f[c]=np.sign(np.dot(w[c], test_input))
			g[c]=np.dot(w[c], test_input)
		guess=max(g, key=g.get)	
		if test_y[k][0]==guess:
			accuracy.append(1)
		else:
			accuracy.append(0)	
		
	accuracy_rate=float(sum(accuracy))/float(len(accuracy))
	error_rate=1-accuracy_rate
	print 'Perceptron V0 error rate=', error_rate
	return error_rate