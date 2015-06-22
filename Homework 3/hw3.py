import os
from collections import Counter
import numpy as np
import re
import stemming
import stemming.porter
import numpy.matlib
import scipy.stats
import math
import random
import numpy.linalg


def bag_of_words(m=100):
	ham=0
	spam=1

	folders=np.array([os.path.join(os.getcwd(),'enron1\\ham'), os.path.join(os.getcwd(),'enron1\\spam')])
	n=min(len(os.listdir(folders[ham])),len(os.listdir(folders[spam])))

	indices = np.random.permutation(n)
	training_idx, test_idx = indices[:m], indices[m:]


	test_x=np.hstack([np.asarray(os.listdir(folders[ham]))[test_idx],np.asarray(os.listdir(folders[spam]))[test_idx]])
	test_y=np.hstack([np.ones(len(test_idx)), np.ones(len(test_idx))])


	training_x=np.hstack([np.asarray(os.listdir(folders[ham]))[training_idx],np.asarray(os.listdir(folders[spam]))[training_idx]])
	training_y=np.hstack([np.zeros(len(training_idx)), np.ones(len(training_idx))])
	
	x=[]
	dictionary=[]
	counter={}

	for i in training_x:
		with open(os.path.join(folders[training_y[np.where(training_x==i)[0][0]]], i)) as f:
			word_list=[stemming.porter.stem(word.lower()) for word in re.sub("[^a-zA-Z]"," ", f.read()).split()]
			x.append(word_list)
			counter[np.where(training_x==i)[0][0]]=scipy.stats.itemfreq(word_list)		
			for w in word_list:
				stem_word=w
				dictionary.append(stem_word)

	x_array=np.array(x)
	distinct_dict=np.unique(dictionary)
	word_bag=np.matlib.repmat(np.zeros(len(distinct_dict)),len(x),1)

	for i in x_array:
		idx=x.index(i)
		for i in counter[idx]:
			word_bag[idx][np.where(distinct_dict==i[0])]=i[1]

	return word_bag, training_y, test_x, test_y, distinct_dict

	

def classify_new_email(i,m):
	beta, test_x, test_y, distinct_dict=log_reg(m)
	folders=np.array([os.path.join(os.getcwd(),'enron1\\ham'), os.path.join(os.getcwd(),'enron1\\spam')])
	ham=0
	spam=1

	x=[]
	counter={}

	if i.find('ham.txt')>0:
		test_y=ham
		with open(os.path.join(folders[ham], i)) as f:
			word_list=[stemming.porter.stem(word.lower()) for word in re.sub("[^a-zA-Z]"," ", f.read()).split()]
			counter[np.where(i==i)[0][0]]=scipy.stats.itemfreq(word_list)	
	elif i.find('spam.txt')>0:
		test_y=spam
		with open(os.path.join(folders[spam], i)) as f:
			word_list=[stemming.porter.stem(word.lower()) for word in re.sub("[^a-zA-Z]"," ", f.read()).split()]
			counter[np.where(i==i)[0][0]]=scipy.stats.itemfreq(word_list)	
	
	x_array=np.array(word_list)		
	word_bag=np.matlib.repmat(np.zeros(len(distinct_dict)),1,1)


	for j in x_array:
		c=np.where(counter[0][:,0]==j)[0][0]
		if j in distinct_dict:
			b=np.where(distinct_dict==j)[0][0]
			word_bag[0,np.where(distinct_dict==j)[0][0]]=counter[0][c,1]

	b_0=np.ones((1,1))
	word_bag=np.hstack((word_bag,b_0))

	p=np.exp(np.dot(word_bag,beta))/(1+np.exp(np.dot(word_bag,beta)))
	if p >= 0.5:
		guess=1
	elif p < 0.5:
		guess=0	

	return p, guess, test_y

def f(word_bag,training_y,beta):
	a=np.dot(word_bag, beta)
	b=(a.transpose()*training_y).transpose()
	l=sum(b-np.log(1+np.exp(a)))
	return l

def f_prime(word_bag,training_y,beta):
	p=np.exp(np.dot(word_bag,beta))/(1+np.exp(np.dot(word_bag,beta)))
	dldb=np.dot(word_bag.transpose(), np.subtract(training_y,p.transpose()).transpose())
	return dldb

def f_double_prime(word_bag,training_y,beta):
	p=np.exp(np.dot(word_bag,beta))/(1+np.exp(np.dot(word_bag,beta)))
	a=word_bag.transpose()
	b=word_bag*np.multiply(p,(1-p))
	dl2db=np.negative(np.dot(a,b))
	return dl2db


def log_reg(m=10):
	word_bag, training_y, test_x, test_y, distinct_dict=bag_of_words(m)
	b_0=np.ones((2*m,1))
	word_bag=np.hstack((word_bag,b_0))
	iterations=0
	made_changes=True
	max_iterations=50

	beta=np.repeat(0, word_bag.shape[1]).reshape(word_bag.shape[1], 1)
	old_f=f(word_bag,training_y,beta)
	fdouble=f_double_prime(word_bag,training_y,beta)

	new_beta=np.subtract(beta,np.dot(np.linalg.inv(f_double_prime(word_bag,training_y,beta).transpose()+.001*np.eye(word_bag.shape[1])),f_prime(word_bag,training_y,beta)))
	new_f=f(word_bag,training_y,new_beta)
	rel_change=np.inf
	while made_changes==True and iterations< max_iterations:
		beta=new_beta
		old_f=new_f
		rel_change_old=rel_change

		iterations=iterations+1
		made_changes=False
		new_beta=np.subtract(beta, np.dot(np.linalg.inv(f_double_prime(word_bag,training_y,beta).transpose()+.001*np.eye(word_bag.shape[1])),f_prime(word_bag,training_y,beta)))
		new_f=f(word_bag,training_y,new_beta)
		rel_change=abs(new_f-old_f/old_f)-1
		made_changes=rel_change_old>rel_change
		
	if made_changes==True:
		print 'Did not converge' 
	return beta, test_x, test_y, distinct_dict
		