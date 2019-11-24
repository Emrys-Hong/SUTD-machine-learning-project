import numpy as np
from scipy import optimize,misc

np.random.seed(1234)

def get_x_y(file):
	f=open(file,'r')
	xdistinct=set()
	ydistinct=set()
	x=[]
	y=[]
	xtemp=[]
	ytemp=[]
	lines=[line.split() for line in f]
	for i in range(len(lines)):
		if len(lines[i])>0:
			xdistinct.add(lines[i][0])
			ydistinct.add(lines[i][1])
	xdistinct=list(xdistinct)
	# print ydistinct
	ydistinct=list(ydistinct)
	ydistinct=['START']+ydistinct+['STOP']
	# print ydistinct
	ymap=dict(enumerate(ydistinct))
	# print ymap
	yindex=list(ymap.keys())
	# print yindex
	# print ymap

	for i in range(len(lines)):
		if len(lines[i])>0:
			xtemp.append(lines[i][0])
			for j in list(ymap.keys()):
				if lines[i][1]==ymap[j]:
					ytemp.append(j)
			# ytemp.append(int({k:v for k,v in ymap.items() if v==lines[i][1]}))
		else:
			x.append(xtemp)
			y.append(ytemp)
			xtemp=[]
			ytemp=[]
	

	return x,y,xdistinct,yindex,ymap

def get_feature_functions_update(xdistinct, ydistinct):
	f = {}
	feature_index = 0
	for y1 in range(len(ydistinct)):
		for y2 in range(len(ydistinct)):
			f[str(y1) + " " + str(y2)] = feature_index
			feature_index = feature_index + 1
	for x in xdistinct:
		# print x 
		for y in ydistinct:
			f[str(y) + " " + x] = feature_index
			feature_index = feature_index + 1
	# print f
	return f

def get_feature_function_matrix(x_vec,ydistinct,feature_functions_update):
	# print type(x_vec)
	# print "feature length: " , len(feature_functions_update)
	feature_fns=np.zeros((len(x_vec)+1,len(ydistinct),len(ydistinct),len(feature_functions_update)),dtype=int)
	for i in range(len(x_vec)+1):
		for y1 in range(len(ydistinct)):
			for y2 in range(len(ydistinct)):
				if y1 == len(ydistinct)-1 or  y2 == 0:
					continue
				feature_fns[i,y1,y2][feature_functions_update[str(y1) + " " + str(y2)]] = 1
				if i > 0 and i <= len(x_vec):
					try:
						feature_fns[i-1,y1,y2][feature_functions_update[str(y2) + " " + x_vec[i-1] ]] = 1
					except:
						continue
				# for f in range(len(feature_functions_update)):
				# 	feature_fns[i,y1,y2,f]=feature_functions_update[f](ydistinct[y1],ydistinct[y2],x_vec,i)
				# print "i: ", i, " y1: ", y1, " y2: ", y2, " " , feature_fns[i,y1,y2], np.sum(feature_fns[i,y1,y2])
	# print feature_fns.shape
	return feature_fns

def forward(log_M):
	alpha=np.NINF*np.ones((log_M.shape[0]+1,log_M.shape[1]),dtype=float)
	alpha[0][0]=0
	for i in range(1,log_M.shape[0]+1):
		alpha[i]=misc.logsumexp(np.expand_dims(alpha[i-1],axis=1)+log_M[i-1],axis=0)
		# alpha[i]=np.dot(alpha[i-1], M[i-1])
		# alpha[i]=np.log(np.sum(np.exp(np.expand_dims(alpha[i-1],axis=1) + np.log(M[i-1]))))
	# print alpha
	return alpha

	# alpha=np.zeros((log_M.shape[0]+1,log_M.shape[1]),dtype=float)
	# # alpha[0][0]=0
	# alpha[0][1:]=np.NINF
	# for i in range(1,log_M.shape[0]+1):
	# 	for y1 in range(log_M.shape[1]):
	# 		for y2 in range(log_M.shape[1]):
	# 			# if y1==M.shape[1]-1 or y2==0:
	# 			# 	continue
	# 			alpha[i,y2]+=np.exp(alpha[i-1,y1]+log_M[i-1,y1,y2])
	# 	# alpha[i,1:]=np.log(alpha[i,1:]-1)
	# 	alpha[i]=np.log(alpha[i])
	# 	# alpha[i,0]=np.NINF
	# # print alpha
	# return alpha

def backward(log_M):
	beta=np.NINF*np.ones((log_M.shape[0]+1,log_M.shape[1]),dtype=float)
	beta[-1][-1]=0
	for i in range(log_M.shape[0]-1,-1,-1):
		beta[i]=misc.logsumexp(log_M[i]+np.expand_dims(beta[i+1],axis=0),axis=1)
		# beta[i]=np.dot(M[i],beta[i+1])
		# beta[i]=np.log(np.sum(np.exp(beta[i+1] + np.log(M[i]))))
	# print beta
	return beta

	# beta=np.zeros((log_M.shape[0]+1,log_M.shape[1]),dtype=float)
	# # beta[M.shape[0]-1][M.shape[1]-1]=0
	# beta[-1][:-1]=np.NINF
	# for i in range(log_M.shape[0]-1,-1,-1):
	# 	for y1 in range(log_M.shape[1]):
	# 		for y2 in range(log_M.shape[1]):
	# 			# if y1==M.shape[1]-1 or y2==0:
	# 			# 	continue
	# 			beta[i,y1]+=np.exp(beta[i+1,y2]+log_M[i,y1,y2])
	# 	# beta[i,:-1]=np.log(beta[i,:-1]-1)
	# 	# beta[i,-1]=np.NINF
	# 	beta[i]=np.log(beta[i])
	# # print beta
	# return beta


def neg_likelihood(theta,feature_functions,x_vec,y_vec,ydistinct):
	likelihood=0
	derivative=np.zeros(len(theta))
	final_exp_features=np.zeros(len(theta))
	emp_features=np.zeros(len(theta))

	for x,y in zip(x_vec,y_vec):
		n=len(x)

		feature_fns=get_feature_function_matrix(x,ydistinct,feature_functions)
		log_M=np.dot(feature_fns,theta)
		# print feature_fns.shape
		# print log_M

		alpha=forward(log_M)
		beta=backward(log_M)

		# print 'alpha',alpha
		# print 'beta',beta
		# print 'beta',beta[0][0]
		# print 'alpha',alpha[-1][-1]

		# p=0
		# for i in range(len(alpha[3])):
		# 	p+=np.exp(alpha[3][i])*np.exp(beta[3][i])/np.exp(beta[0][0])
		# print p

		log_Z=beta[0][0]

		likelihood+=log_M[0,0, y[0]]
		emp_features += feature_fns[0, 0, y[0]]
		
		for y2 in ydistinct[1:]:
			exp_features=np.exp(alpha[0,0] + log_M[0,0,y2] + beta[1,y2] - log_Z) * feature_fns[0,0,y2]
			final_exp_features += exp_features

		for i in range(1,len(x)):
			emp_features+=feature_fns[i, y[i-1], y[i]]
			likelihood+=log_M[i,y[i-1], y[i]]

			for y1 in ydistinct[1:-1]:
				for y2 in ydistinct[1:-1]:
					exp_features= np.exp(alpha[i,y1] + log_M[i,y1,y2] + beta[i+1,y2] - log_Z) * feature_fns[i,y1,y2]
					final_exp_features += exp_features

		# for y1 in ydistinct[:-1]:
		# 	exp_features= np.exp(alpha[n,-1] + log_M[n,y1,-1] + beta[n+1,-1] - log_Z) * feature_fns[n,y1,-1]
		# 	final_exp_features += exp_features
		
		# emp_features+=feature_fns[n, y[-1],-1]

		# likelihood+=log_M[n,y[-1],-1]

		likelihood-=log_Z
	# likelihood-=np.sum(theta**2/2/0.64)
	# print emp_features
	# print exp_features
	derivative=emp_features-final_exp_features#-theta/0.64
	print(('likelihood',-likelihood))
	# print 'derivative',-derivative
	return -likelihood,-derivative

def train(feature_functions,x_vec,y_vec,ydistinct):
	theta=np.random.randn(len(feature_functions))

	f =lambda theta: neg_likelihood(theta,feature_functions,x_vec,y_vec,ydistinct)
	# # print theta.shape
	result=optimize.fmin_l_bfgs_b(f,theta)
	# print 'result',result
	theta=result[0]


	# for i in range(1000):	
	# 	# print theta
	# 	l,d=neg_likelihood(theta,feature_functions,x_vec,y_vec,ydistinct)
	# 	theta-=0.05*d

	return theta

def viterbi(x_vec,ydistinct,feature_functions,theta):
	# x_vec=x_vec[:1]
	ybest_sequence_all=[]
	print((len(x_vec)))
	for x in x_vec:
		feature_fns=get_feature_function_matrix(x,ydistinct,feature_functions)
		log_M=np.dot(feature_fns,theta)
		# print len(x)
		# print log_M
		# print feature_fns.shape
		delta=np.zeros((len(x),len(ydistinct)),dtype=np.float)
		psi=np.zeros((len(x),len(ydistinct)),dtype=np.int)
		# print theta
		for y in range(1,len(ydistinct)-1):
			delta[0,y]=log_M[0,0,y]
			# delta[0,y]=exp_dot_product(feature_functions,'START',ydistinct[y],x,0,theta)
		for i in range(1,len(x)):
			for y1 in range(1,len(ydistinct)-1):
				# print delta[i-1,:]+np.dot(feature_fns[i,:,y1],theta)
				delta[i,y1]=np.max(delta[i-1,1:-1]+log_M[i,1:-1,y1])
				psi[i,y1]=np.argmax(delta[i-1,1:-1]+log_M[i,1:-1,y1])+1

		ybestreverse=[]
		# ybestreverse.append(max(delta[len(x_vec)-1,:]))
		# print ydistinct
		# delta_stop=np.max(delta[len(x)-1,1:-1]+log_M[len(x),1:-1,-1])
		# print delta
		psi_stop=np.argmax(delta[len(x)-1,1:-1]+log_M[len(x),1:-1,-1])+1
		# print psi

		# ybest=ydistinct[-1]
		ybest=psi_stop
		# print ybest
		for i in range(len(x)-1,-1,-1):
			# print psi[i,ybest]
			ybestreverse.append(ybest)
			ybest=psi[i,ybest]
		# print ybestreverse
		# print len(ybestreverse)
		ybest_sequence=[]
		for i in range(len(ybestreverse)-1,-1,-1):
			ybest_sequence.append(ydistinct[int(ybestreverse[i])])
		ybest_sequence_all.append(ybest_sequence)
		# print ybest_sequence_all
	return ybest_sequence_all


def get_actual_labels(y_sequence,ymap):
	y_actual_all=[]
	for i in y_sequence:
		y_actual=[]
		for y in i:
			y_actual.append(ymap[y])
		y_actual_all.append(y_actual)
	return y_actual_all

def accuracy(y_actual,y_result):
	correct=0
	total=0
	for i in range(len(y_actual)):
		for j in range(len(y_actual[i])):
			total+=1
			if y_actual[i][j]==y_result[i][j]:
				correct+=1
	accuracy=1.0*correct/total
	return accuracy



x_vec,y_vec,xdistinct,ydistinct,ymap=get_x_y('/Users/pengfei/Github/sutd-machine-learning-project/dataset/EN/train')
# print ymap
# print ydistinct,y_vec
# print y_vec
# print len(x),len(y),len(xdistinct),len(ydistinct)
# x_vec,y_vec=get_sentence('train copy')
# print x_vec
feature_functions=get_feature_functions_update(xdistinct,ydistinct)
# feature_functions1=get_feature_functions(xdistinct,ydistinct)
print ('training started')
theta=train(feature_functions,x_vec,y_vec,ydistinct)
print ('training finished')
# print ymap

x_vec,y_vec,_,_,ymap=get_x_y('/Users/pengfei/Github/sutd-machine-learning-project/dataset/EN/train')
# feature_functions=get_feature_functions_update(xdistinct,ydistinct)
# feature_functions=get_feature_functions(xdistinct,ydistinct)
y_result=viterbi(x_vec,ydistinct,feature_functions,theta)
# print y_result
print(('Accuracy on test data:',accuracy(y_vec,y_result) ))
y_result=get_actual_labels(y_result,ymap)
print(('Predicted:',y_result))

x_vec,y_vec,_,_,ymap=get_x_y('train.data')
# feature_functions=get_feature_functions_update(xdistinct,ydistinct)
# feature_functions=get_feature_functions(xdistinct,ydistinct)
y_result=viterbi(x_vec,ydistinct,feature_functions,theta)
# print y_result
print('Accuracy on train data:',accuracy(y_vec,y_result))
y_result=get_actual_labels(y_result,ymap)
print('Predicted:',y_result)