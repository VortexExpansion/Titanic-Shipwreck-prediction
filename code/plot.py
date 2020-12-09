import matplotlib.pyplot as plt
from train_model import get_train

train_data = get_train()
print(train_data.head())
print(train_data['Survived'].value_counts().plot(kind="bar"))
plt.show()