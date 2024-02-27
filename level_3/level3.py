import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier


df =  pd.read_csv('training_data.csv', names=['dance', 'message'])
index=[]
for i in range(len(df['dance'])):
    count = len(df['dance'][i])
    if count !=30:
        index.append(i)

df.drop(index, inplace=True)


df = df[df.message <2]

'''plt.hist(df['message'], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('messages')
plt.ylabel('number of messages')
plt.title('histogram of classification')
plt.show()

msg = np.array(df['message'])
patterns = []
for i in range(5):
    patterns.append(np.count_nonzero(msg == i))

for i in range(len(patterns)):
    print(patterns[i])'''
    


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


print("-------------- SVM ---------------")

svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print(accuracy_score(y_test,y_pred))


print("-------------- MLP -----------------")

mlp = MLPClassifier(hidden_layer_sizes=(100,100), activation='tanh',
                        solver = 'adam', alpha = 0.0001, learning_rate='constant', 
                        max_iter = 400, learning_rate_init = 0.01).fit(x_train,y_train)

print(accuracy_score(y_test,mlp.predict(x_test)))

