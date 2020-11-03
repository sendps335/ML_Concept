#Log_Loss_Practice
#Classic Cost Function
import numpy as np
import math

def Adam_Optimizer(theta,X,y,alpha,beta,epsilon):
    theta_gd,delta=gradient_descent(theta,X,y,alpha,pow_t)
    delta_shape=delta.shape
    s_delta=beta*delta+(1-beta)*([i*i for i in delta].reshape(delta_shape[0],delta_shape[1]))
    net_delta=[]
    for i in range(0,delta_shape[0]):
        net_delta_row=[]
        for j in range(0,delta_shape[1]):
            delta[i][j]=delta[i][j]/(1-beta**pow_t)
            s_delta[i][j]=s_delta[i][j]/(1-beta**pow_t)
            k=delta[i][j]/((s_delta[i][j]-epsilon)**0.5)
            net_delta_row.append(k)
        net_delta.append(net_delta_row)
    theta=theta-net_delta
    return theta

def momentum_gradient_descent(theta,X,y,alpha,beta):
    theta_gd,delta=gradient_descent(theta,X,y,alpha)
    delta_shape=delta.shape
    delta_new=(1-beta)*delta
    theta=theta-delta_new
    return theta
    
def RMS_gradient_descent(theta,X,y,alpha,beta,epsilon):
    theta_gd,delta=gradient_descent(theta,X,y,alpha)
    delta_shape=delta.shape
    s_delta=beta*delta+(1-beta)*([i*i for i in delta].reshape(delta_shape[0],delta_shape[1]))
    net_delta=[]
    for i in range(0,delta_shape[0]):
        net_delta_row=[]
        for j in range(0,delta_shape[1]):
            k=(delta[i][j])/((s_delta[i][j]-epsilon)**(0.5))
            net_delta_row,append(k)
        net_delta.append(net_delta_row)
    theta=theta-net_delta
    return theta

def gradient_descent(theta,X,y,alpha):
    m=len(X)
    n=len(X[0])
    delta=(alpha/m)*np.dot(X.T,np.dot(X,theta)-y)
    theta=theta-delta
    return theta,delta

def linear_regression(y_pred,y_true):
    J=0
    m=len(y_pred)
    for i in range(0,m):
        sq_error_su=(y_pred[i]-y_true[i])**2
        J=J+sq_error_su
    J=J/(2*m)
    return J

def Logistic_regression(y_pred,y_true):
    J=0
    m=len(y_pred)
    for i in range(0,m):
        sq_error_su=-1*(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
        J=J+sq_error_su
    J=J/(2*m)
    return J

def linear_prediction(X,y_true,alpha,no_of_iters):
    cost_err_dict=dict()
    cost_err_pos=dict()
    theta=np.random.randn(len(X[0]),1)
    for i in range(0,no_of_iters):
        y_pred=np.dot(X,theta)
        cost_function=linear_regression(y_pred,y_true)
        cost_err_dict[i+1]=cost_function
        cost_err_pos[cost_function]=theta
        theta,delta=gradient_descent(theta,X,y,alpha)
    kk=min(cost_err_dict.values)
    theta_min=cost_err_pos[kk]
    return theta_min,kk

def cost_function(y_pred,y_true,choice):
    if(choice==1):
        return linear_regression(y_pred,y_true)
    elif(choice==2):
        return logistic_regression(y_pred,y_true)
    return -1