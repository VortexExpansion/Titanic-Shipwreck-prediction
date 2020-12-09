import matplotlib.pyplot as plt
from train_model import get_train

import seaborn as sns
sns.set(font_scale=1)

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


# Optimization, more visual, use this later

# g = sns.catplot(x="Sex", y="Survived", col="Pclass",
#                     data=train_data, saturation=.5,
#                     kind="bar", ci=None, aspect=.6)
# (g.set_axis_labels("", "Survival Rate")
#     .set_xticklabels(["Men", "Women"])
#     .set_titles("{col_name} {col_var}")
#     .set(ylim=(0, 1))
#     .despine(left=True))  
# plt.subplots_adjust(top=0.8)
# g.fig.suptitle('Survival according to passenger class')
# plt.show()