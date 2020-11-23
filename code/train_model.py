import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import csv
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
exposure = train.head()

def write_to_new_file(filename):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["PassengerId","Survived"])
        ID=892
        for loop_var in range(len(prediction)):
            writer.writerow([ID,prediction[loop_var]])
            ID+=1

# for c in test.columns:
#     print(c,100*test[c].isnull().sum()/len(test))
    #To locate the missing values in the data table as percentages

train['Age'] = train['Age'].fillna(train['Age'].mean()) #replacing null values by mean ages
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test["Fare"].fillna(test['Age'].mean())
#print(exposure)


dependencies_sex = train[['Sex', 'Survived']].groupby(['Sex'],as_index=False).mean()
dependencies_Pclass = train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean()
dependencies_parch = train[['Parch', 'Survived']].groupby(['Parch'],as_index=False).mean()

# print(dependencies_Pclass) #class 1 more likely
# print(dependencies_sex) #female more likely
# print(dependencies_parch) #parch 1,2,3 more likely

train=train.drop(['PassengerId','Name','Cabin','Embarked','Ticket'],axis=1) #removing all redundant columns
test=test.drop(['PassengerId','Name','Cabin','Embarked','Ticket'],axis=1)
#print(train.head())

Sex_mapping={'female':1,'male':0}
train.Sex=[Sex_mapping[gender] for gender in train.Sex] #Replacing string feature by int for model fitting
test.Sex=[Sex_mapping[gender] for gender in test.Sex]

#print(train.head())

#Fitting the model

X=train.drop(['Survived'],axis=1)
y=train.Survived
#print(X.head())
#print(test.head())


'''

# ----------------    Decision Tree Classifier   : This model got an accuracy of 73% on kaggle

model_1=DecisionTreeClassifier(random_state=1)
model_1.fit(X,y)
prediction = model_1.predict(test)
print(prediction)

#percent_1=metrics.accuracy_score(prediction,y)
#print('Model accuracy of Decision Tree Classifier = {}%'.format(100*percent_1))

write_to_new_file('DTClassifier.csv')
  

'''

'''
# ----------------- Random Forest Classifier : This model got an accuracy of 74% on kaggle

model_2 = RandomForestClassifier(random_state=1)
model_2.fit(X,y)
prediction = model_2.predict(test)
print(prediction)

write_to_new_file('RandomForestClassifier.csv')
'''