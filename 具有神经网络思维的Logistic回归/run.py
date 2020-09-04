import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset

train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=load_dataset()
m_train=train_set_y.shape[1]#训练集图片数量
m_test=test_set_y.shape[1]#test图片数量
num_px=train_set_x_orig.shape[1]#训练、测试里面图片的宽度和高度64x64

#训练集维度降低并转置
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
#测试集维度降低并转置
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s

#初始化w和b
def initialize_with_zeros(dim):
    #创建(dim,1)的0向量，将b初始化0
    w=np.zeros(shape=(dim,1))
    b=0
    assert(w.shape==(dim,1))
    #b的类型是float或者是int
    assert(isinstance(b,float) or isinstance(b,int))
    return (w,b)

#计算成本函数、梯度
def propagate(w,b,X,Y):
    m=X.shape[1]

    #正向传播
    A=sigmoid(np.dot(w.T,X)+b)
    cost=(-1/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))

    #反向传播
    dz=A-Y
    dw=(1/m)*np.dot(X,dz.T)
    db=(1/m)*np.sum(dz)

    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost=np.squeeze(cost)
    assert(cost.shape==())

    grads={

        'dw':dw,
        'db':db
    }
    return (grads,cost)

#梯度下降优化w、b
def optimize(w,b,X,Y,num_interations,learning_rate,print_cost=False):
    #num_interations 优化循环的迭代次数
    #print_cost每一百步打印损失值
    costs=[]
    for i in range(num_interations):
        grads,cost=propagate(w,b,X,Y)
        dw=grads['dw']
        db=grads['db']

        w=w-learning_rate*dw
        b=b-learning_rate*db

        if i%100==0:
            costs.append(cost)

        if (print_cost)and(i%100==0):
            print('迭代次数：%i，误差值：%f'%(i,cost))

    params={
        'dw':dw,
        'db':db
    }
    return (params,grads,costs)

#预测函数Y_prediction  0.5为判断界限
def predict(w,b,X):
    m=X.shape[1]
    Y_prediction=np.zeros((1,m))
    w=w.reshape(X.shape[0],1)
    A=sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        Y_prediction[0,i]=1 if A[0,i]>0.5 else 0
    assert (Y_prediction.shape==(1,m))
    return Y_prediction

def model(X_train,Y_train,X_test,Y_test,num_interations=2000
          ,learning_rate=0.5,print_cost=False):
    '''
        参数：
        X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
        X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
        num_iterations  - 表示用于优化参数的迭代次数的超参数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost  - 设置为true以每100次迭代打印成本
    '''
    w,b=initialize_with_zeros(X_train.shape[0])
    parameters,grads,costs=optimize(w,b,X_train,Y_train,num_interations, learning_rate , print_cost)
    w,b=parameters['dw'],parameters['db']

    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)


    d={
        'cost':costs,
        'Y_prediction_test':Y_prediction_test,
        'Y_prediction_train':Y_prediction_train,
        'w':w,
        'b':b,
        'learning_rate':learning_rate,
        'num_interations':num_interations
    }
    return d

d=model(train_set_x,train_set_y,test_set_x,test_set_y,num_interations=2000,learning_rate = 0.005, print_cost=True)
costs=np.squeeze(d['cost'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('interations')
plt.title('learning_rate:'+str(d['learning_rate']))
plt.show()