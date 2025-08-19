import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
data=pd.read_csv("ex4.csv")
df=pd.DataFrame(data)
x=df[['study hour','attendance']]
y=df['result']
clf=DecisionTreeClassifier(criterion='entropy',random_state=0)
clf.fit(x,y)
plt.figure(figsize=(4,4))
plot_tree(clf,feature_names=['study hour','attendance'],class_names=['fail','pass'],filled=True)
plt.show()
new=[[5,85]]
pred=clf.predict(new)
print("prediction for new student:","1"if pred[0]==1 else "0")
