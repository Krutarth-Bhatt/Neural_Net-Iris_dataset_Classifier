
# This is a Neural Net for classification of Iris Dataset.
# I have divided the dataset into 90 random training examples and 30 random testing examples.
# First layer consists of 4 neurons and the second layer consists of 100 neurons with activation "relu" 
# and the third layer has 3 neurons with activation "softmax".
# I am getting 100% accuracy on TestData and about 96% accuracy on TrainData.
# The training and testing files are attached.

# We have Changed the labels:
#                            1 --> setosa
#                            2 --> versicolor
#                            3 --> virginica


# Developer: Krutarth Bhatt---------(NIRMA UNIVERSITY)




"""The code looks BIG just because I have made functions for training and testing and intializing weights and
   biases. Any No of layers and their neurons can be added from the last part of the code."""






import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


# activation functions::
def tanh(x, derivative=False):
    if (derivative == True):
        return (1 - (x ** 2))
    x = np.array(x)
    y = np.tanh(x)
    return y

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    #print(e_x.shape)
    e_x_sum = np.sum(e_x, axis=0, keepdims=True)
    #print(e_x_sum.shape)
    return np.divide(e_x, e_x_sum)



def softmax_test(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    #print(e_x.shape)
    e_x_sum = np.sum(e_x, axis=1, keepdims=True)
    #print(e_x_sum.shape)
    return np.divide(e_x, e_x_sum)


def relu(x, derivative=False):
    if (derivative == True):
        for i in range(0, len(x)):
            for k in range(len(x[i])):
                if x[i][k] > 0:
                    x[i][k] = 1
                else:
                    x[i][k] = 0
        return x
    for i in range(0, len(x)):
        for k in range(0, len(x[i])):
            if x[i][k] > 0:
                pass  # do nothing since it would be effectively replacing x with x
            else:
                x[i][k] = 0
    return x

def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

d = pd.read_csv('/home/kuku/Documents/train_iris.csv').as_matrix()
x_train = np.array(d[:,:4])
x_train = x_train.T
y_train = np.array([d[:,4]])
y_train = y_train.astype(int)

q = pd.read_csv('/home/kuku/Documents/test_iris.csv').as_matrix()
x_test = np.array(q[:,:4])
x_test = x_test.T
y_test = np.array([q[:,4]])
y_test = y_test.astype(int)


# Intializing parameters::
def params_in(layer_dims):
    para = {}
    l = len(layer_dims)
    for i in range(1,l):
        para["w"+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1]) * 0.01
        para["b"+str(i)] = np.zeros((layer_dims[i],1))
    return para

#training::
def train(para, acti, z, acti_type, iterations, alpha, train_no):
    tic = time.time()
    jwb = []
    itr = []
    ctr = 1
    
    for i in range(iterations):
        # for prop::
        l = len(acti_type)
        for j in range(1,l+1):
            z['z'+str(j)] = np.dot(para['w'+str(j)],acti['a'+str(j-1)]) + para['b'+str(j)]
            #activate::
            if acti_type[j-1]== "tanh":
                acti['a'+str(j)] = tanh(z["z"+str(j)])
            elif acti_type[j-1]== "softmax":
                acti['a'+str(j)] = softmax(z["z"+str(j)])
            elif acti_type[j-1]== "relu":
                acti['a'+str(j)] = relu(z["z"+str(j)])
            elif acti_type[j-1]== "sigmoid":
                acti['a'+str(j)] = sigmoid(z["z"+str(j)])
                
        #calculating Cost Function::    
        lost = np.array(acti['a'+str(l)] + 0.01)
        lost = np.log(lost)
        li = 0
        for j in range(train_no):
            li += -lost[y_train[0,j]-1,j]
        jwb.append(li/train_no)
        itr.append(ctr)
        ctr+=1       
                
        #back prop for last layer:: 
        dz ={}
        dz[str(l)] = acti['a'+str(l)]
        for k in range(train_no):
            dz[str(l)][y_train[0,k]-1,k] -= 1
        dw = np.dot(dz[str(l)], acti['a'+str(l-1)].T)/train_no
        db = np.sum(dz[str(l)], axis=1, keepdims = True)/train_no
        para['w'+str(l)] = para['w'+str(l)] - (alpha*dw)
        para['b'+str(l)] = para['b'+str(l)] - (alpha*db)
        
        #back prop for other layers::
        for k in range(l-1,0,-1):
            if acti_type[k-1]=='tanh': g = tanh(z['z'+str(k)],True)
            elif acti_type[k-1]=='relu': g = relu(z['z'+str(k)],True)
            elif acti_type[k-1]=='sigmoid': g = sigmoid(z['z'+str(k)],True)
            #calculating Derivatives::    
            dz[str(k)] = (np.dot(para['w'+str(k+1)].T, dz[str(k+1)]) * g)
            dw = np.dot(dz[str(k)], acti['a'+str(k-1)].T)/train_no
            db = np.sum(dz[str(k)], axis=1, keepdims = True)/train_no
            #Gradient dececnt::
            para['w'+str(k)] = para['w'+str(k)] - (alpha*dw)
            para['b'+str(k)] = para['b'+str(k)] - (alpha*db)
    toc = time.time()
    print("training time: "+str((toc-tic)/60)+" min")
    
    
    plt.figure(figsize = (12, 9))
    plt.title("Change in Cost Function over training iterations::", fontsize=20)
    plt.xlabel("ith iteration", fontsize=22)
    plt.ylabel("cost", fontsize=22)
    plt.plot(itr, jwb)
            
#training::            
def test(para, acti_type, test_no):
    accuracy_score = 0
    wrong = 0
    for i in range(test_no):
        l = len(acti_type)
        x = x_test[:,i]
        x = np.array([x])
        z_test = {}
        acti_test = { 'a0':x}
        for j in range(1,l+1):
            z_test['z'+str(j)] = np.dot(acti_test['a'+str(j-1)], para['w'+str(j)].T) + para['b'+str(j)].T
            #activate::
            if acti_type[j-1]== "tanh":
                acti_test['a'+str(j)] = tanh(z_test["z"+str(j)])
            elif acti_type[j-1]== "softmax":
                acti_test['a'+str(j)] = softmax_test(z_test["z"+str(j)])
            elif acti_type[j-1]== "relu":
                acti_test['a'+str(j)] = relu(z_test["z"+str(j)])
            elif acti_type[j-1]== "sigmoid":
                acti_test['a'+str(j)] = sigmoid(z_test["z"+str(j)]) 
        q = -1
        ans=0
        for j in range(3):
            #print(acti_test['a'+str(l)][0,j])
            if q < acti_test['a'+str(l)][0,j]: 
                q = acti_test['a'+str(l)][0,j]
                ans=j+1
        if ans==y_test[0,i]: accuracy_score+=1    
    
    print("Accuracy: "+str((accuracy_score/test_no)*100)+" %")
    print("No. of wrong ans: "+str(test_no-accuracy_score))
    

    
    
""" You can set No of layers and their No of neurons and their activations from the below code:: """


list = [4, 100, 3]        #List consisting No of neurons in its respective indexed layer 
para = params_in(list)    #para is a dictionary that has all weights and biases 
param_intial = para

acti_type = ["relu", "softmax"]
acti = {'a0': x_train}    #acti is a dictionary consisiting of activations of every layer as they get calculated.
z = {}
train(para, acti, z, acti_type, 175, 0.054, 119)
test(para, acti_type, 29)






