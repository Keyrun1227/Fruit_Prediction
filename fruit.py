import pandas as pd
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
import pickle




df=pd.read_csv('fruits.csv')

array=df.values
x=array[:,0:5]
y=array[:,6]
validation_size=0.20
seed=6
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=validation_size,random_state=seed)
scoring='accuracy'

classifier = GaussianNB()  
classifier.fit(x_train, y_train) 

pickle.dump(classifier,open('fruit.pkl','wb')) 