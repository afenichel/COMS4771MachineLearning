import scipy
from scipy import io
from scipy import ndimage
from scipy import stats
import numpy as np
import math

#m is the training size
#c is the class - 0,1,2,3,4,5,6,7,8,9
#k is the index of the test data

#part i. classifier per class c=[0,1,2,3,4,5,6,7,8,9]
def hw_k(c, T, training_x, training_y):
	training_ybin=training_y.astype(int)	
	training_ybin[training_ybin!=c]=np.negative(1)
	training_ybin[training_ybin==c]=1
	d=np.size(training_x,1)
	n=np.size(training_x,0)
	alpha=np.array(np.zeros(n))

	for t in range(T):
		x_dot=np.power(1+np.dot(training_x, training_x.transpose()),5)
		ay=np.multiply(alpha,training_ybin.transpose()).transpose()
		ayx_dot=ay*x_dot
		ks=np.sign(np.sum(ayx_dot, axis=0))
		compare_y=training_ybin.transpose()-ks
		idx_x, idx_y= np.where(compare_y!=0)
		alpha[idx_y]=alpha[idx_y]+1

	ay=np.multiply(alpha,training_ybin.transpose()).transpose()		
	return ay

	
#part i. test for one input x_test[k] on all classes c
def classifier_k(m,k,T):
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
		ay=hw_k(c, T, training_x, training_y)
		x_dot_test=np.power(1+np.dot(training_x, test_input.transpose()),5)	
		ayx_dot=ay.transpose()*x_dot_test
		g[c]=np.sum(ayx_dot)
		f[c]=np.sign(g)
	
	print 'Perceptron Kernel output:'
	print 'guess=',  max(g, key=g.get)
	print 'y_actual=',test_y[k]

	return max(g, key=g.get), test_y[k]


#part ii. loop through all testing inputs
def test_k(m,T, training_x, training_y, test_x, test_y):
	y_list=[i for i in range(10)]
	y_list=np.random.permutation(y_list)
	g={}
	f={}
	ay={}

	for c in y_list:
		ay[c]=hw_k(c, T, training_x, training_y)
	return ay
	


def accuracy_k(m, T):
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
	
	ay=test_k(m, T, training_x, training_y, test_x, test_y)
	accuracy=[]
	y_list=[i for i in range(10)]
	y_list=np.random.permutation(y_list)


	for k in range(1000):
		f={}
		g={}
		test_input=test_x[k,:]
		for c in y_list:
			x_dot_test=np.power(1+np.dot(training_x, test_input.transpose()),5)	
			ayx_dot=ay[c].transpose()*x_dot_test
			g[c]=np.sum(ayx_dot)
			f[c]=np.sign(g)
		

		guess=max(g, key=g.get)	
		if test_y[k][0]==guess:
			accuracy.append(1)
		else:
			accuracy.append(0)	


	accuracy_rate=float(sum(accuracy))/float(len(accuracy))
	error_rate=1-accuracy_rate
	print 'error rate=', error_rate
	return error_rate