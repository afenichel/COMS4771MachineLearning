import numpy as np
import numpy.linalg
import numpy.matlib
import matplotlib.pyplot as plt

cities=['BOS','NYC','DC','MIA','CHI','SEA','SF','LA','DEN']
A=np.array([[0,206,429,1504,963,2976,3095,2979,1949],[0,0,233,1308,802,2815,2934,2786,1771],[0,0,0,1075,671,2684,2799,2631,1616],[0,0,0,0,1329,3273,3053,2687,2037],[0,0,0,0,0,2013,2142,2054,996],[0,0,0,0,0,0,808, 1131,1307],[0,0,0,0,0,0,0,379,1235],[0,0,0,0,0,0,0,0,1059],[0,0,0,0,0,0,0,0,0]])

D=A+A.transpose()
n=D.shape[0]



x=np.arange(1,19).reshape(9,2)

x1=x[:,0].reshape(n,1)
x2=x[:,1].reshape(n,1)

dist=np.sqrt(np.power(x1-np.matlib.repmat(x1.transpose(),n,1),2)+np.power(x2-np.matlib.repmat(x2.transpose(),n,1),2))


f1=2*(dist-D)*(x1-np.matlib.repmat(x1.transpose(),n,1))/dist
f2=2*(dist-D)*(x2-np.matlib.repmat(x2.transpose(),n,1))/dist
f_prime=np.vstack((sum(np.nan_to_num(f1.transpose())), sum(np.nan_to_num(f2.transpose()))))

x_new=x-.01*f_prime.transpose()
x1_new=x_new[:,0].reshape(n,1)
x2_new=x_new[:,1].reshape(n,1)
dist_new=np.sqrt(np.power(x1_new-np.matlib.repmat(x1_new.transpose(),n,1),2)+np.power(x2_new-np.matlib.repmat(x2_new.transpose(),n,1),2))


f_x=sum(sum(np.power(dist-D,2)))
f_x_new=sum(sum(np.power(dist_new-D,2)))
iteration=0
while f_x>f_x_new and iteration < 5000:
	iteration=iteration+1
	x=x_new
	x1=x[:,0].reshape(n,1)
	x2=x[:,1].reshape(n,1)
	dist=np.sqrt(np.power(x1-np.matlib.repmat(x1.transpose(),n,1),2)+np.power(x2-np.matlib.repmat(x2.transpose(),n,1),2))
	f1=2*(dist-D)*(x1-np.matlib.repmat(x1.transpose(),n,1))/dist
	f2=2*(dist-D)*(x2-np.matlib.repmat(x2.transpose(),n,1))/dist
	f_prime=np.vstack((sum(np.nan_to_num(f1.transpose())), sum(np.nan_to_num(f2.transpose()))))
	x_new=x-.01*f_prime.transpose()
	x1_new=x_new[:,0].reshape(n,1)
	x2_new=x_new[:,1].reshape(n,1)
	dist_new=np.sqrt(np.power(x1_new-np.matlib.repmat(x1_new.transpose(),n,1),2)+np.power(x2_new-np.matlib.repmat(x2_new.transpose(),n,1),2))
	f_x=sum(sum(np.power(dist-D,2)))
	f_x_new=sum(sum(np.power(dist_new-D,2)))

x=x


print 'stress=',np.sqrt(sum(sum(np.power(dist-D,2)))/sum(sum(np.power(D,2))))

print 'optimal setting of coordinates of x are :', x

plt.plot(x[:,0], x[:,1],'bo')
for i in cities:
	plt.annotate(i, xy=x[cities.index(i),:])
plt.show()