import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('Exp 2.csv')
x=df[['Bedrooms','Size','Age','Zipcode']]
y=df['Selling Price']
ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(),['Zipcode'])],remainder='passthrough')
xen=ct.fit_transform(x)
xtr,xte,ytr,yte=train_test_split(xen,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(xtr,ytr)
ypr=model.predict(xte)
ypr
coefficients=model.coef_
intercept=model.intercept_
print("Coefficients:",coefficients)
print("Intercept:",intercept)
plt.figure(figsize=(8,6))
sns.scatterplot(x=yte,y=ypr,color='blue',s=100)
plt.plot([min(yte),max(yte)],[min(yte),max(yte)],'r--')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted House Price")
plt.grid(True)
plt.tight_layout()
plt.show()
sns.heatmap(x.corr(),annot=True,cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
