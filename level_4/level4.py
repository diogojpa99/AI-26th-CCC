import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras

df =  pd.read_csv('training_data.csv', names=['dance', 'message'])
index=[]

for i in range(len(df['dance'])):
    count = len(df['dance'][i])
    if count !=30:
        index.append(i)

df.drop(index, inplace=True)

dic = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}
names = ['dance', 'message']  
df_nr = pd.DataFrame(names)  

dance_numb=[]
for dance in df['dance']:
    aux = []
    for i in dance:
        aux.append(dic[i])
    dance_numb.append(aux)
data = pd.DataFrame(dance_numb)

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:].values,
                                                    df.iloc[:,(len(df.columns)-1)].values,
                                                    test_size=0.1,shuffle=True)


'''print("-------------- SVM ---------------")



svm = SVC(C=1.0,kernel='poly')
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print(accuracy_score(y_test,y_pred))


print("-------------- MLP -----------------")

mlp = MLPClassifier(hidden_layer_sizes=(30,100,50,25), activation='tanh',
                        solver = 'adam', alpha = 0.0001, learning_rate='constant', 
                        max_iter = 400, learning_rate_init = 0.01).fit(x_train,y_train)
y_pred = mlp.predict(x_test)
print(accuracy_score(y_test,y_pred))


print("-------------- NB ---------------")

svm = MultinomialNB()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print(accuracy_score(y_test,y_pred))

print("-------------- KNN ---------------")

svm = KNeighborsClassifier(n_neighbors=10)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print(accuracy_score(y_test,y_pred))'''


print("-------------- RNN ---------------")

x = df['dance'].values.reshape((df['dance'].values.shape[0], 30, 1)) # (5850, 33, 1)
model = tf.keras.Sequential()
model.add(keras.layers.LSTM(50, activation='relu', input_shape=(30, 1)))
model.add(keras.layers.Dense(4))
model.add(keras.layers.Dense(4))
model.add(keras.layers.Dense(4))
#early_stopping = EarlyStopping(monitor='val_loss', patience=42)  
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
model.fit(x, df['message'], epochs=200, verbose=1, validation_split = 0.2)
