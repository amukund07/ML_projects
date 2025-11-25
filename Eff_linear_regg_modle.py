# Multvariable efficent LRM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing

# Import data set
medical_df= pd.read_csv("medical.csv")

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

def rems(targets,predictions):
    return  np.sqrt(np.mean(np.square(targets-predictions)))

ages=medical_df.age
bmis=medical_df.bmi
childrens=medical_df.children
sex_column = medical_df.sex_code
smoker_column = medical_df.smoker_code
northwests = medical_df.northwest
southeasts = medical_df.southeast
southwests = medical_df.southwest

def try_para(p,q,r,s,t,u,v,w,b):
    targets=medical_df.charges
    predictions= estimates(ages,bmis,childrens,sex_column,smoker_column, northwests, southeasts, southwests,p,q,r,s,t,u,v,w,b)
    plt.figure(figsize=(10,6))
    plt.scatter(ages, targets, alpha=0.5)
    plt.scatter(ages, predictions, alpha=0.5)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Actual', 'Predicted'])

    loss=rems(targets,predictions)
    print("loss", loss)

# training model
model= LinearRegression()
inputs = medical_df[['age',
                     'bmi',
                     'children',
                     'sex_code',
                     'smoker_code',
                     'northwest',
                     'southeast',
                     'southwest']]

targets=medical_df.charges
model.fit(inputs,targets)

try_para(model.coef_[0],model.coef_[1],model.coef_[2],model.coef_[3],model.coef_[4],model.coef_[5],model.coef_[6],model.coef_[7],model.intercept_)

# Getting data on weights 
input_cols = inputs.columns 
weights_df = pd.DataFrame({
    'feature': np.append(input_cols, "intercept"),
    'weight': np.append(model.coef_, model.intercept_)
})
print(weights_df)

plt.show()

