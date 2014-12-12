import numpy as np
import pylab

t=[30,50,50]

grad=[0,0,0]

x=list()
x.append([1,7,2])
x.append([1,3,6])
x.append([1,2,2])
x.append([1,4,9])
x.append([1,3,12])
x.append([1,5,22])

y=list()
y.append(5)
y.append(17)
y.append(8)
y.append(18)
y.append(28)
y.append(30)

dev=list()

def gradient_descent(alpha, x, y, ep, max_iter):
    converged = False
    iter=0
    m=len(x) # number of samples
    n=len(t) # number of theta

    # total error, J(theta)
    J=0.5*sum([(t[0]*x[i][0]+t[1]*x[i][1]+t[2]*x[i][2]-y[i])**2 for i in range(m)])

    # Iterate Loop
    while not converged:
        #for each training sample, compute the gradient (d/d_theta J(theta))
        for k in range(n):
            grad[k]=sum([(t[0]*x[i][0]+t[1]*x[i][1]+t[2]*x[i][2]-y[i])*x[i][k] for i in range(m)])
        for k in range(n):
            t[k]-=alpha*grad[k]

        e = 0.5*sum([(t[0]*x[i][0]+t[1]*x[i][1]+t[2]*x[i][2]-y[i])**2 for i in range(m)])
        print(e)
        dev.append(e) # obtain dev for plotting

        if abs(J-e) <= ep:
            converged= True

        J = e #update error
        iter +=1   # update iter

        if iter == max_iter:
            converged = True

    return t

if __name__=='__main__':

    gradient_descent(0.001, x, y, 0.1, 10000)

    print('theta0 = %s theta1 = %s theta2 = %s') %(t[0], t[1], t[2])

    #plot
    pylab.plot(dev)
    pylab.show()
    print "Done!"

    
