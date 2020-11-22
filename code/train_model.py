import pandas as pd

train = pd.read_csv("./data/train.csv")
exposure = train.head()

#for c in train.columns:
    #print(c,100*train[c].isnull().sum()/len(train))
    #To locate the missing values in the data table as percentages

train['Age'] = train['Age'].fillna(train['Age'].mean()) #replacing null values by mean ages

#print(exposure)

dependencies_sex = train[['Sex', 'Survived']].groupby(['Sex'],as_index=False).mean()
dependencies_Pclass = train[['Pclass', 'Survived']].groupby(['Pclass'],as_index=False).mean()
dependencies_parch = train[['Parch', 'Survived']].groupby(['Parch'],as_index=False).mean()

print(dependencies_Pclass) #class 1 more likely
print(dependencies_sex) #female more likely
print(dependencies_parch) #parch 1,2,3 more likely








