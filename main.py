import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE= "model.pkl"
PIPELINE_FILE="pipeline.pkl"

def build_pipeline(num_attribs,cat_attribs):
    num_pipeline=Pipeline(
        [
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
        ]
    )

    cat_pipeline=Pipeline(
        [
            ("onehot",OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    full_pipeline=ColumnTransformer([
        ("num",num_pipeline,num_attribs),
        ("cat",cat_pipeline,cat_attribs)
    ])
    
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    housing=pd.read_csv("housing.csv")

    housing['income_cat']=pd.cut(housing["median_income"],bins=[0.0,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

    split=StratifiedShuffleSplit(n_splits=1,random_state=42,test_size=0.2)

    for train_index,test_index in split.split(housing,housing['income_cat']):
        housing.iloc[test_index].drop('income_cat',axis=1).to_csv("input.csv",index=False)
        housing= housing.iloc[train_index].drop('income_cat',axis=1)
        
        
    housing_labels=housing["median_house_value"].copy() 
    housing_features=housing.drop("median_house_value",axis=1)
    
    num_attribs=housing_features.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attribs=["ocean_proximity"]
    
    pipeline=build_pipeline(num_attribs,cat_attribs)
    # print(housing_features)
    housing_prep=pipeline.fit_transform(housing_features)
    # print(housing_prep)
    
    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prep,housing_labels)
    
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model is trained, Congratulations")
    
else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
    
    inputdata=pd.read_csv('input.csv')
    transformed_input=pipeline.transform(inputdata)
    predictions=model.predict(transformed_input)
    inputdata['median_house_value']=predictions
    
    inputdata.to_csv('output.csv',index=False)
    print("Complete")