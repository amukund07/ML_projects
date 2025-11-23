# Age,bmi,children
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

medical_df = pd.read_csv('medical.csv')

non_smoker_df=medical_df[medical_df.smoker=='no']

def estimate(age,bmi,children,u,v,w,b):
    return age*u+bmi*v+children*w+b

def rems(target,prediction):
    return np.sqrt(np.mean(np.square(target-prediction)))

ages=non_smoker_df.age
bmis=non_smoker_df.bmi
childs=non_smoker_df.children

def try_para(u,v,w,b):
    targets=non_smoker_df.charges
    predictions= estimate(ages,bmis,childs,u,v,w,b)
    plt.figure(figsize=(10,6))
    plt.scatter(ages, targets, alpha=0.5)
    plt.scatter(ages, predictions, alpha=0.5)
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.legend(['Actual', 'Predicted'])

    loss=rems(targets,predictions)
    print("loss", loss)


model= LinearRegression()
inputs=non_smoker_df[['age','bmi','children']]
targets=non_smoker_df.charges
model.fit(inputs, targets)


try_para(model.coef_[0],model.coef_[1],model.coef_[2],model.intercept_)
plt.show()
