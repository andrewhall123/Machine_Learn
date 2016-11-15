import numpy as np
from sympy import *
from sklearn.datasets.samples_generator import make_regression 
from scipy import stats
import matplotlib.pyplot as plt

def gradient_descent(x,y,alpha,max_iter=1000,ep=0.0001):
	converged=False
	iters=0
	err=0
	m=x.shape[0]#number of samples
	t0=np.random.random(x.shape[1])
	t1=np.random.random(x.shape[1])
	
	J=sum([(t0+t1*x[i]-y[i])**2 for i in range(m)])
	
	#iterate loop
	while not converged:
		grad0=1.0/m*sum([(t0+t1*x[i]-y[i]) for i in range(m)])
		grad1=1.0/m*sum([(t0+t1*x[i]-y[i])*x[i] for i in range(m)])
		
		temp0=t0-alpha*grad0
		temp1=t1-alpha*grad1
		
		t0=temp0
		t1=temp1
		
		#mean squared error
		e=sum([(t0+t1*x[i]-y[i])**2 for i in range(m)])
		
		if abs(J-e)<=ep:
			print("Converged\n")
			converged=True
		
		#update j and iters
		J=e
		iters+=1
		if iters==max_iter:
			print("max reached")
			converged=True
	return t0,t1
		
x, y = make_regression(n_samples=100, n_features=1, n_informative=1, 
                        random_state=0, noise=35) 
                        
print(x,"\n\n\n",y,"\n\n",x.shape[1] )
print( 'x.shape = %s y.shape = %s' %(x.shape, y.shape))

alpha = 0.01 # learning rate
ep = 0.01 # convergence criteria

# call gredient decent, and get intercept(=theta0) and slope(=theta1)
theta0, theta1 = gradient_descent(x, y, alpha,ep=ep)
print (('theta0 = %s theta1 = %s') %(theta0, theta1) )

#check with scipy linear regression
slope,inter,r_val,p_val,slop_std=stats.linregress(x[:,0],y)

for i in range(x.shape[0]):
	y_predict=theta0+theta1*x

plt.plot(x,y,'o')
plt.plot(x,y_predict,'k--')
plt.show()




