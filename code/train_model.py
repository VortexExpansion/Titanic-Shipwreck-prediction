import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


train = pd.read_csv("./data/train.csv")
exposure = train.head()

# for c in train.columns:
#     print(c,100*train[c].isnull().sum()/len(train))
#     To locate the missing values in the data table as percentages

train['Age'] = train['Age'].fillna(train['Age'].mean()) #replacing null values by mean ages

#print(exposure)

dependencies_sex = train[['Sex', 'Survived']].groupby(['Sex'],as_index=False).mean()
dependencies_Pclass = train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean()
dependencies_parch = train[['Parch', 'Survived']].groupby(['Parch'],as_index=False).mean()

# print(dependencies_Pclass) #class 1 more likely
# print(dependencies_sex) #female more likely
# print(dependencies_parch) #parch 1,2,3 more likely

train=train.drop(['PassengerId','Name','Cabin','Embarked','Ticket'],axis=1) #removing all redundant columns
#print(train.head())

Sex_mapping={'female':1,'male':0}
train.Sex=[Sex_mapping[gender] for gender in train.Sex] #Replacing string feature by int for model fitting

#print(train.head())

#Fitting the model

X=train.drop(['Survived'],axis=1)
y=train.Survived
#print(X.head())

# Decision Tree Classifier

model_1=DecisionTreeClassifier(random_state=1)
model_1.fit(X,y)
prediction = model_1.predict(X)
#print(prediction)
percent_1=metrics.accuracy_score(prediction,y)
print('Model accuracy of Decision Tree Classifier = {}%'.format(100*percent_1))





