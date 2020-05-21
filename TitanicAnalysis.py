import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

titanic_data=pd.read_csv("/Users/kuldipsingh/Desktop/titanicdataset.csv")
print(titanic_data.head(10))

print("# of passengers in original data:"+str(len(titanic_data.index)))

### ANALYZING DATA
print("="*10)
print("ANALYZING DATA")
print("="*10)
sns.countplot(x="Survived",data=titanic_data)
plt.show()

sns.countplot(x="Survived",hue="Sex",data=titanic_data)
plt.show()


sns.countplot(x="Survived",hue="Pclass",data=titanic_data)
plt.show()

titanic_data["Age"].plot.hist()
plt.show()

titanic_data["Fare"].plot.hist(bins=20,figsize=(10,5))
plt.show()

titanic_data.info()

sns.countplot(x="SibSp",data=titanic_data)
plt.show()

sns.countplot(x="Parch",data=titanic_data)
splt.show()

sns.countplot(x="Cabin",data=titanic_data)
plt.show()

sns.countplot(x="Embarked",data=titanic_data)
plt.show()

##DATA WRANGLING
print("="*10)
print("DATA WRANGLING")
print("="*10)
print(titanic_data.isnull())

print(titanic_data.isnull().sum())

sns.heatmap(titanic_data.isnull(),yticklabels=False,cmap='viridis')
plt.show()

sns.boxplot(x="Pclass",y="Age",data=titanic_data)
plt.show()

print(titanic_data.head(5))

titanic_data.drop("Cabin",axis=1,inplace=True)

print(titanic_data.head(5))

titanic_data.dropna(inplace=True)

sns.heatmap(titanic_data.isnull(),yticklabels=False,cbar=False)
plt.show()

print(titanic_data.isnull().sum())

sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
print(sex.head(5))

embark=pd.get_dummies(titanic_data['Embarked'],drop_first=True)
print(embark.head(5))


Pc1=pd.get_dummies(titanic_data['Pclass'],drop_first=True)
print(Pc1.head(5))

titanic_data=pd.concat([titanic_data,sex,embark,Pc1],axis=1)
print(titanic_data.head(5))

titanic_data.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)
print(titanic_data.head(5))

titanic_data.drop('Pclass',axis=1,inplace=True)
print(titanic_data.head(5))

##TRAIN DATA
print("="*10)
print("TRAIN DATA")
print("="*10)
X=titanic_data.drop("Survived",axis=1)
y=titanic_data["Survived"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train,X_test,y_train,y_test=train_test_split(StandardScaler().fit_transform(X),y,test_size=0.3,random_state=1)

from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
#print(logmodel)
print(logmodel.fit(X_train,y_train))


predictions=logmodel.predict(X_test)

from sklearn.metrics import classification_report
classification_report(y_test,predictions)
#or
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))

