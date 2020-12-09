import matplotlib.pyplot as plt
from train_model import get_train

train_data = get_train()
print(train_data.head())

print(train_data['Survived'].value_counts().plot(kind="bar"))
plt.title('Survived')
plt.show()

print(train_data[['Age']].plot(kind='hist',bins=[0,20,40,60,80,100],rwidth=0.8))
plt.title('Age wise distribution')
plt.show()

print(train_data['Pclass'].value_counts().plot(kind="bar"))
plt.title('Pclass')
plt.show()

print(train_data['Sex'].value_counts().plot(kind="bar"))
plt.title('Sex')
plt.show()