"""
构建具有单隐藏层的2类分类神经网络。
使用具有非线性激活功能激活函数，例如tanh。
计算交叉熵损失（损失函数）。
实现向前和向后传播。
"""
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

'''
testCases：提供了一些测试示例来评估函数的正确性
planar_utils ：提供了在这个任务中使用的各种有用的功能
'''

np.random.seed(1)   #伪随机数
X,Y=load_planar_dataset()
#y=0 红色，y=1蓝色
plt.scatter(X[0,:],X[1,:],c=Y,s=40,cmap=plt.cm.Spectral)#绘制散点图
shape_X=X.shape
shape_Y=Y.shape
m=Y.shape[1]#训练集里面的数据
'''
print('X的维度：'+str(shape_X))
print('Y的维度：'+str(shape_Y))
print('数据集里面的数据：'+str(m)+'个')
'''

def layer_sizes(X,Y):
    """
        参数：
         X - 输入数据集,维度为（输入的数量，训练/测试的数量）
         Y - 标签，维度为（输出的数量，训练/测试数量）

        返回：
         n_x - 输入层的数量
         n_h - 隐藏层的数量
         n_y - 输出层的数量
    """
    n_x=X.shape[0]
    n_h=4
    n_y=Y.shape[0]

    return(n_x,n_h,n_y)

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    w1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros(shape=(n_h,1))
    w2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros(shape=(n_y,1))

    assert(w1.shape==(n_h,n_x))
    assert(b1.shape==(n_h,1))
    assert(w2.shape==(n_y,n_h))
    assert (b2.shape==(n_y,1))

    parameters={
        'w1':w1,
        'b1':b1,
        'w2':w2,
        'b2':b2
    }
    return parameters

def forward_propagation(X,parameters):
    w1=parameters['w1']
    b1=parameters['b1']
    w2=parameters['w2']
    b2=parameters['b2']

    #前向传播
    z1=np.dot(w1,X)+b1
    a1=np.tanh(z1)
    z2=np.dot(w2,a1)+b2
    a2=sigmoid(z2)

    assert(a2.shape==(1,X.shape[1]))

    cache={
        'Z1':z1,
        'A1':a1,
        'Z2':z2,
        'A2':a2
    }
    return(a2,cache)

def compute_cost(A2,Y,parameters):
    m=Y.shape[1]
    w1=parameters['w1']
    w2=parameters['w2']

    #计算成本
    logprobs=np.multiply(np.log2(A2),Y)+np.multiply(np.log2(1-A2),1-Y)
    cost=-np.sum(logprobs)/m
    cost=float(np.squeeze(cost))

    assert(isinstance(cost,float))
    return cost

def backward_propagation(parameters,cache,X,Y):
    m=X.shape[1]

    w1=parameters['w1']
    w2=parameters['w2']

    A1=cache['A1']
    A2=cache['A2']

    dz2=A2-Y
    dw2=(1/m)*np.dot(dz2,A1.T)
    db2=(1/m)*np.sum(dz2,axis=1,keepdims=True)
    dz1=np.multiply(np.dot(w2.T,dz2),1-np.power(A1,2))
    dw1=(1/m)*np.dot(dz1,X.T)
    db1=(1/m)*np.sum(dz1,axis=1,keepdims=True)

    grads={
        'dw1':dw1,
        'dw2':dw2,
        'db1':db1,
        'db2':db2
    }
    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    w1,w2=parameters['w1'],parameters['w2']
    b1,b2=parameters['b1'],parameters['b2']

    db1,db2=grads['db1'],grads['db2']
    dw1,dw2=grads['dw1'],grads['dw2']

    w1=w1-learning_rate*dw1
    w2=w2-learning_rate*dw2
    b1=b1-learning_rate*db1
    b2=b2-learning_rate*db2

    parameters={
        'w1':w1,
        'b1':b1,
        'w2':w2,
        'b2':b2
    }
    return parameters

def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    np.random.seed(3)
    n_x=layer_sizes(X,Y)[0]
    n_y=layer_sizes(X,Y)[2]

    parameters=initialize_parameters(n_x,n_h,n_y)
    w1=parameters['w1']
    w2=parameters['w2']
    b1=parameters['b1']
    b2=parameters['b2']

    for i in range(num_iterations):
        A2,cache=forward_propagation(X,parameters)
        cost=compute_cost(A2,Y,parameters)
        grads=backward_propagation(parameters,cache,X,Y)
        parameters=update_parameters(parameters,grads,learning_rate=0.5)

        if print_cost:
            if i%100==0:
                print('第',i,'次循环，成本为'+str(cost))
    return parameters

def predict(parameters,X):
    A2,cache=forward_propagation(X,parameters)
    predictions=np.round(A2)#四舍五入
    return predictions

parameters=nn_model(X,Y,n_h=4,num_iterations=10000,print_cost=True)
plt.show()