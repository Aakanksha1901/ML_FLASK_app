import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle


data = {
    'Temperature': [20,22,24,26,28,30],
    'Sales': [150,165,180,195,210,225]
}

df = pd.DataFrame(data)
print(df)

X = df[['Temperature']]
y = df['Sales']


model = LinearRegression()
model.fit(X, y)
print("Model trained successfully")

# save model
pickle.dump(model, open('model.pkl', 'wb'))
print("Model saved")