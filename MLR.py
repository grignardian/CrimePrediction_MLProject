#MLR for crimerate vs socioeconomic factors district lvl 2011
#model training
import pandas as pd
df=pd.read_excel("data.xlsx") #fill1
print(df.info())
print(df.describe())
X=df[["A","B","C"]] #fill2
y=df["crimerate"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=50)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

#model accuracy measures-r2,mse,coeff&intercept,feature ranking
from sklearn.metrics import r2_score,mean_squared_error
y_pred=model.predict(X_test)
#variance- shld be closer to 1
print("R^2=",r2_score(y_test,y_pred))
#mean sq err shld be lowest
print("MSE=",mean_squared_error(y_test,y_pred))
#coeff and intercept
print("Coefficients:",model.coef_)
print("Intercept:",model.intercept_)
#feature ranking
corr_target=df.corr()["crimerate"].drop("crimerate")
print(corr_target.sort_values(ascending=False))

#graphs
#correlation heatmap
import matplotlib.pyplot as plt
plt.imshow(df.corr(),cmap="coolwarm")
plt.colorbar()
plt.show()
#scatter plot
plt.scatter(df["A"],df["crimerate"]) #fill3
plt.show()
plt.scatter(df["B"],df["crimerate"]) #fill4
plt.show()
plt.scatter(df["C"],df["crimerate"]) #fill5
plt.show()
#pred vs actual
plt.scatter(y_test,y_pred)
plt.plot([y.min(),y.max()],[y.min(), y.max()],'r--')
plt.show()
#coeff bar chart
plt.bar(X.columns,model.coef_)
plt.show()
#boxplot
plt.boxplot(df[["A","B","C","crimerate"]].values)
plt.xticks(range(1,5),["A","B","C","crimerate"])
plt.show()
#histogram
plt.hist(df["crimerate"],bins=20)
plt.show()
#violin plot
plt.violinplot(df[["A","B","C","crimerate"]].values)
plt.xticks(range(1,5),["A","B","C","crimerate"])
plt.show()
#scatter matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df[["A","B","C","crimerate"]],figsize=(8,8))
plt.show()
#corr matrix
print(df.corr())
