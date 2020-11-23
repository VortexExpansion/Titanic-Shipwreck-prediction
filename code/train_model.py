import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def test_accuracy(prediction,y):

    '''
    Func to check the fitting of model on the test data set
    Parameters :
    prediction = values of the output that are predicted by the model
    y = actual values stated in test data  
    '''

    percent = metrics.accuracy_score(prediction,y)
    print('Accuracy of the chosen model = {}%'.format(100*percent))



def write_to_new_file(filename):

    '''
    Creates a new csv file in the output folder and writes the predictions of the model
    Parameter: 
    filename : name of the new csv file to be created 
    ''' 

    path="./output/"+filename
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["PassengerId","Survived"])
        ID=892
        for loop_var in range(len(prediction)):
            writer.writerow([ID,prediction[loop_var]])
            ID+=1



def train_model(model,filename='example.csv'):

    '''
    Func to train the model on the given test data set. 
    Parameters : 
    model : name of the model to be applied
    filename : string which stores the name of newfile to be created. Deafault argument = test.csv
    '''

    model.fit(X,y)
    global prediction
    prediction = model.predict(test)
    print(prediction)
    write_to_new_file(filename)

def check_missing_values(test):

    '''
    Func to print the percentage of missing values in the dataset
    Parameter :
    test : Given dataset
    '''
    for c in test.columns:
        print(c,100*test[c].isnull().sum()/len(test))

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
exposure = train.head()

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
# ----------------    Decision Tree Classifier   : This model got an accuracy of 73.6% on kaggle

model_1=DecisionTreeClassifier(random_state=1)
train_model(model_1,'test1.csv')

# ----------------- Random Forest Classifier : This model got an accuracy of 74.4% on kaggle

model_2 = RandomForestClassifier(random_state=2)
train_model(model_2,'test3.csv')

# ----------------- Logistic Regression : This model got an accuracy of 75.8% on kaggle

model_3 = LogisticRegression(random_state=3)
train_model(model_3,'LogisticRegression.csv')

'''