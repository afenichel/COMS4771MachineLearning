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
def hw_v2(c, T, training_x, training_y):
	training_ybin=training_y.astype(int)	
	training_ybin[training_ybin!=c]=np.negative(1)
	training_ybin[training_ybin==c]=1
	d=np.size(training_x,1)
	n=np.size(training_x,0)

	w=np.array([np.zeros(d)])
	cc=np.array([0])
	k=1
	for t in range(T):
		i=(t+1)%(n+1)

		if training_ybin[i-1]*np.dot(w[k-1], training_x[i-1,:].transpose()) <= 0:
			w=np.vstack([w, w[k-1]+training_ybin[i-1]*training_x[i-1,:]])
			cc=np.vstack([cc, 1])
			k=k+1
		else: 
			cc[k-1]=cc[k-1]+1

	
	return w, cc



#part i. test for one input x_test[k] on all classes c
def classifier_v2(m,k,T):
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
	s={}
	q={}
	test_input=test_x[k,:]

	for c in y_list:
		w, cc=hw_v2(c, T, training_x, training_y)
		s[c]=np.sign(np.dot(w, test_input.transpose()))
		q[c]=np.multiply(cc.transpose(),s[c])
		g[c]=sum(q[c][0])
		f[c]=np.sign(g)

	print 'Perceptron V2 output:'	
	print 'guess=',  max(g, key=g.get)
	print 'y_actual=',test_y[k]

	return max(g, key=g.get), test_y[k]



#part ii. loop through all testing inputs
def test_v2(m, T, training_x, training_y, test_x, test_y):
	y_list=[i for i in range(10)]
	y_list=np.random.permutation(y_list)
	cc={}
	w={}
	
	for c in y_list:
		w[c], cc[c]=hw_v2(c, T, training_x, training_y)
	return w, cc
		


def accuracy_v2(m, T):
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

	w, cc=test_v2(m, T, training_x, training_y, test_x, test_y)
	accuracy=[]
	y_list=[i for i in range(10)]
	y_list=np.random.permutation(y_list)

	for k in range(1000):
		f={}
		g={}
		q={}
		s={}
		test_input=test_x[k,:]
		for c in y_list:
			s[c]=np.sign(np.dot(w[c], test_input.transpose()))
			q[c]=np.multiply(cc[c].transpose(),s[c])
			g[c]=sum(q[c][0])
			f[c]=np.sign(g)

		guess=max(g, key=g.get)	
		if test_y[k][0]==guess:
			accuracy.append(1)
		else:
			accuracy.append(0)	

	accuracy_rate=float(sum(accuracy))/float(len(accuracy))
	error_rate=1-accuracy_rate
	print 'Perceptron V2 error rate=', error_rate
	return error_rate