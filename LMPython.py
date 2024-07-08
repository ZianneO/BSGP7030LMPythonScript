#!/usr/bin/env python
# coding: utf-8

# # Linear Modeling in Python

# In[1]:



from sklearn.linear_model import LinearRegression


from sklearn.metrics import r2_score 


# ## Reading in csv file


df = pd.read_csv("regrex1.csv")



df.head()



df.head


# # Data in Scatter Plot


plt.scatter(df['x'],df['y'], color= "hotpink")


# # Linear Model


X= df[['x']]
y= df['y']


# In[14]:


model = LinearRegression()
model.fit(X,y)


# # Obtaining coefficients

# ### Intercept and slope

# In[15]:


r_sq = model.score(X,y)
print(f"Intercept: {model.intercept_}")
print(f"Coefficients (Slope): {model.coef_}")



slope = model.coef_[0]


# ### R^2



r2 = r2_score(y, y_pred)




print(f"Coefficient of Determination (R-squared): {r_sq}")




y_pred = model.predict(df[['x']])




print("Predictions:", y_pred)


# # Linear Model with Slope, R^2, and axes labels



plt.figure(figsize=(8, 6))
plt.scatter(df['x'], y, color='pink', label='Original data')
plt.plot(df['x'], y_pred, color='purple', linewidth=2, label='Linear model line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot with Linear Regression Line')
plt.legend()
plt.grid(True)
plt.text(0.95, 0.05, f'slope: {slope:.2f}\nR^2: {r2:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=12)
plt.show()













