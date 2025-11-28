# Import to read dataset
import numpy as np
import pandas as pd
# Import to make graphs
import matplotlib.pyplot as plt
# Import for model traning
from sklearn.linear_model import LinearRegression 
# Import for Standardization of the weights
from sklearn import preprocessing   
from sklearn.preprocessing import StandardScaler        
# Import for split train test
from sklearn.model_selection import train_test_split

# Import data set
medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
from urllib.request import urlretrieve
urlretrieve(medical_charges_url, 'medical.csv')
medical_df = pd.read_csv('medical.csv')

# Sorting data using 1 and 0
smoker_codes = {'no': 0, 'yes': 1}
medical_df['smoker_code'] = medical_df.smoker.map(smoker_codes)

sex_codes={"male":0,"female":1}
medical_df["sex_code"]= medical_df.sex.map(sex_codes)

# One hot encoding 
enc=preprocessing.OneHotEncoder()
enc.fit(medical_df[["region"]])
one_hot = enc.transform(medical_df[['region']]).toarray()
medical_df[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot

print(medical_df.head(5))
print(medical_df.describe())
# age,bmi,children,sex_code,smoker_code, northwest, southeast, southwest
# In this we only take one in binary encoding like either male or female coz m+f=1
# In this we only take three in one hot encoding coz ne+nw+se+sf=1

def estimates(age,bmi,children,sex_code,smoker_code, northwest, southeast, southwest,p,q,r,s,t,u,v,w,b):
    return age*p+bmi*q+children*r+sex_code*s+smoker_code*t+northwest*u+southeast*v+southwest*w+b

def rmse(targets,predictions):
    return  np.sqrt(np.mean(np.square(targets-predictions)))

# Setting up input 
ages=medical_df.age
bmis=medical_df.bmi
childrens=medical_df.children
sex_column = medical_df.sex_code
smoker_column = medical_df.smoker_code
northwests = medical_df.northwest
southeasts = medical_df.southeast
southwests = medical_df.southwest
# Standardization of the weights
# Fitting on scale
numeric_cols = ['age', 'bmi', 'children']
scalar= StandardScaler()
scalar.fit(medical_df[numeric_cols])
print(scalar.mean_)
print(scalar.var_)

# Standardization
scalar_inputs=scalar.transform(medical_df[numeric_cols])
print(scalar_inputs)

# Combined with the categorical data
cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
categorical_data = medical_df[cat_cols].values

# Training model on the Standarded weights

inputs=np.concatenate((scalar_inputs,categorical_data),axis=1)
targets = medical_df.charges

model= LinearRegression().fit(inputs,targets)

# Generate predictions
predictions = model.predict(inputs)

# Compute loss to evalute the model
loss = rmse(targets, predictions)
print('Loss:', loss)

weights_df = pd.DataFrame({
    'feature': np.append(numeric_cols + cat_cols, 1),
    'weight': np.append(model.coef_, model.intercept_)
})
weights_df.sort_values('weight', ascending=False)
print(weights_df)

# Split train test

inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.1)
# Create and train the model
model = LinearRegression().fit(inputs_train, targets_train)

# Generate predictions
predictions_test = model.predict(inputs_test)

# Compute loss to evalute the model
loss = rmse(targets_test, predictions_test)
print('Test Loss:', loss)

# Generate predictions
predictions_train = model.predict(inputs_train)

# Compute loss to evalute the model
loss = rmse(targets_train, predictions_train)
print('Training Loss:', loss)

