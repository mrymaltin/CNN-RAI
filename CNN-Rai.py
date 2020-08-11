import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
from sklearn.model_selection import StratifiedKFold
import sys
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras import backend as K
from keras.utils import np_utils,  to_categorical
from keras.utils import multi_gpu_model
from sklearn.preprocessing import RobustScaler
from keras.layers.normalization import BatchNormalization
from keras.layers import LeakyReLU
import math
import numpy as np
import tensorflow as tf

# the code modified from "https://github.com/IcarPA-TBlab/MetagenomicDC/blob/master/models/CNN.py"  


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


K.set_session(sess)
#parameters: sys.argv[1] = input dataset as matrix of k-mers
nome_train=sys.argv[1].split(".")[0]				
import math

def convertY2(label_str):
        return int(label_str)+1


def rai(X,k):
    temp_val = 10**(-8)
    b=0

    buyuk_vec=np.empty([len(X),4**(k)])	
    buyuk_vec=np.array(buyuk_vec,dtype=float)    
    for i in X:
        s=sum(i)

        a=0
        for l in i:
            if(s==0):
                s=temp_val
            buyuk_vec[b,a]=(l/s)
            a=a+1
        b=b+1


    kucuk_vec = np.ones([len(buyuk_vec),(4**(k-1))])
    kucuk_vec=np.array(kucuk_vec,dtype=float)    
    for i in range(len(buyuk_vec)):
        for a in range(4**(k-1)):
            kucuk_vec[i,a] = buyuk_vec[i, a*4:a*4+4].sum()

    kucuk_vec=kucuk_vec/kucuk_vec.sum()

    RAI = np.zeros([len(buyuk_vec),(4**k)])

    RAI=np.array(RAI,dtype=float) 
    temp=np.min(buyuk_vec[np.nonzero(buyuk_vec)])
    tempk=np.min(kucuk_vec[np.nonzero(kucuk_vec)])
    for a in range(len(buyuk_vec)):   
        for i in range(4**(k-1)):
            for j in range(4):
                if(kucuk_vec[a, i]==0 ):
                    kucuk_vec[a, i]=tempk
                if(buyuk_vec[a, i*4+j]==0 ):
                    buyuk_vec[a, i*4+j]=temp
                RAI[a,i*4+j] = 11*np.log(buyuk_vec[a, i*4+j]) + np.log(kucuk_vec[a, i])       

    return RAI


def sigmoid(x):
    return 1 / (1 + math.e ** -x)
    
def load_data(file):
    lista=[]
    records= list(open(file, "r"))
    records=records[1:]
    for seq in records:
        elements=seq.split(",")
        level=elements[-1].split("\n")
        classe=elements[0]
        lista.append(convertY2(classe))
    lista=set(lista)
    classes=list(lista)
    X=[]
    Y=[]
    for seq in records:
        elements=seq.split(",")
        X.append(elements[1:-1])
        level=elements[-1].split("\n")
        classe=elements[0]
        Y.append(convertY2(classe))
    X=np.array(X,dtype=float)
    Y=np.array(Y,dtype=int)
    #rai(X, k-mer size)
    X = rai(X,5)
    
    data_max= np.amax(X)
    data_min= np.amin(X)
   

    X=(X-data_min)/(data_max-data_min)


    print(X)


    return X,Y,np.amax(Y)+1,len(X[0])

def create_model(nb_classes,input_length):

       
        model = Sequential()
        model.add(Convolution1D(5,5, border_mode='valid', input_dim=1,input_length=input_length)) #input_dim
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_length=2,border_mode='valid'))
        model.add(Convolution1D(10, 5,border_mode='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_length=2,border_mode='valid'))
        model.add(Flatten())
        ####MLP
        model.add(Dense(500))
        model.add(Activation('relu'))

        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        parallel_model = multi_gpu_model(model, gpus=4)
        parallel_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        return parallel_model


def train_and_evaluate_model (model, X_train, y_train, X_test, y_test,nb_classes,i):
        X2=X_train-np.std(X_train)
        X_train=np.concatenate((X_train,X2), axis=0)
        y_train= np.concatenate((y_train,y_train),axis=0)
        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
        parallel_model = multi_gpu_model(model, gpus=4)

        parallel_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


        parallel_model.fit(X_train , y_train,
          batch_size=20,
          epochs=50,
          validation_data=(X_test, y_test))

      
        return score,acc

name_f="results"


file_n=open(name_f,"w")

if __name__ == "__main__":

    n_folds=10
    seed =7# random seed to repeat result
    np.random.seed(seed)

    X,Y,nb_classes,input_length = load_data("./preprocessing/500r9_5k.txt")
    i=1
    A=[]
    st=0
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)

    for train, test in kfold.split(X, Y):
    

            model = None # Clearing the NN.
            model = create_model(nb_classes,input_length)
            score,acc= train_and_evaluate_model(model, X[train], Y[train], X[test], Y[test],nb_classes,i)

            print('Test score:',score)
            print('Test accuracy: ',acc)

           

