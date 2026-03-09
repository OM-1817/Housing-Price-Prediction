import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
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
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42)
    }

    print("\nModel Comparison (RMSE):")

    for name, model in models.items():
        scores = cross_val_score(
            model,
            housing_prep,
            housing_labels,
            scoring="neg_mean_squared_error",
            cv=10
        )

        rmse_scores = np.sqrt(-scores)
        print(name, "RMSE:", rmse_scores.mean())
    
    param_grid = {
        "n_estimators": [50,100,200],
        "max_features": [4,6,8]
    }

    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error"
    )
    grid_search.fit(housing_prep, housing_labels)
    best_model = grid_search.best_estimator_
    print("\nBest Parameters:", grid_search.best_params_)
    
    predictions = best_model.predict(housing_prep)
    rmse = np.sqrt(mean_squared_error(housing_labels, predictions))
    print("\nFinal Model RMSE:", rmse)
    
    
    feature_importances = best_model.feature_importances_

    features = num_attribs + list(
        pipeline.named_transformers_["cat"]["onehot"].get_feature_names_out(cat_attribs)
    )

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Important Features:")
    print(importance_df.head(10))
    
    
    joblib.dump(best_model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("Model is trained, Congratulations")
    
else:
    best_model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
    
    inputdata=pd.read_csv('input.csv')
    actual=inputdata["median_house_value"]
    features=inputdata.drop('median_house_value',axis=1)
    transformed_input=pipeline.transform(features)
    predictions=best_model.predict(transformed_input)
    # inputdata['median_house_value']=predictions
    
    inputdata.to_csv('output.csv',index=False)
    print("Complete")
    
    
    plt.scatter(actual, predictions)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Housing Prices")

    plt.plot([actual.min(), actual.max()],
            [actual.min(), actual.max()],
            color="red")
    plt.show()
    