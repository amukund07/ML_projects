import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
from urllib.request import urlretrieve
urlretrieve(medical_charges_url, 'medical.csv')
medical_df = pd.read_csv('medical.csv')

# Smoker
smoker_df = medical_df[medical_df.smoker=='yes']

def estimate(age, w, b):
    return w*age + b

def rmse(target, predictions):
    return np.sqrt(np.mean(np.square(target - predictions)))

ages = smoker_df.age

def try_para(w, b):
    target = smoker_df.charges
    predictions = estimate(ages, w, b)

    plt.plot(ages, predictions, 'r', alpha=0.9)
    plt.scatter(ages, target, s=8, alpha=0.9)  
    plt.xlabel('Age')                          
    plt.ylabel('Charges')
    plt.legend(['Prediction', 'Actual'])

    loss = rmse(target, predictions)
    print("loss", loss)

model = LinearRegression()
inputs = smoker_df[['age']]
targets = smoker_df.charges

model.fit(inputs, targets)
print("w ", model.coef_)
print("b ", model.intercept_)
predictions = model.predict(inputs)
print("predictions",predictions)
print("Targets",targets)

try_para(model.coef_, model.intercept_)
plt.show()

