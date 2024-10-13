from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xg
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
m_dir='models'
standard=StandardScaler()
imp_feats=None
def load_model(version):
    filename = f"{m_dir}/model_{version}.pkl"
    if os.path.exists(filename):
        model = joblib.load(filename)
        print("Model",version,"loaded successfully!")
        return model
    else:
        print("Model",version,"not found!")
        return None


model_new=None

def load_dataset(filepath):
    data = pd.read_csv(filepath)
    data = data.drop(columns=['CityID', 'CityName', 'Year'])

    # List of columns to check for outliers
    columns_to_check = ['CurrentPopulation', 'PopulationDensity', 'BirthRate', 
                        'DeathRate', 'ImmigrationRate', 'UnemploymentRate', 
                        'AverageIncome', 'HealthcareAccess', 'CrimeRate']
    
    # Remove outliers
    data = remove_outliers(data, columns_to_check)

    enc_data = pd.get_dummies(data, columns=['EducationLevel'], drop_first=True)
    standard = StandardScaler()
    enc_data[columns_to_check] = standard.fit_transform(enc_data[columns_to_check])
    x = enc_data.drop(columns=['PopulationGrowthRate'], axis=1)
    y = enc_data['PopulationGrowthRate']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test, standard

def train(x_train, y_train, version):
    model_instance = xg.XGBRegressor(objective="reg:squarederror", eval_metric="rmse")

    parameters = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 6, 9]
    }

    param_find_model = GridSearchCV(
        estimator=model_instance, 
        param_grid=parameters, 
        cv=5, 
        scoring='neg_mean_squared_error'
    ) 
    param_find_model.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=0)
    best_model = param_find_model.best_estimator_
    print('Best Hyperparameters:', param_find_model.best_params_)

    if version:
        save_model(best_model, version)

    return best_model

def save_model(model, version):

    if not os.path.exists(m_dir):
        os.makedirs(m_dir)

    mf_name = f"{m_dir}/model_{version}.pkl"
    joblib.dump(model, mf_name)
    print('model saved in file name',mf_name)

def feature_importance(model,x_train, thold=0.01):
    imp=model.feature_importances_
    print(imp)
    imp_feats = x_train.columns[imp > thold]
    print(imp_feats)
    print('no of features important are ',len(imp_feats))
    return x_train[imp_feats],imp_feats

def eval(model, x_test, y_test):
    y_pred = model.predict(x_test)
    ms_error = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Performance:\nMSE: {ms_error:.2f}\nR2 Score: {r2:.2f}")
    return ms_error, r2
    
def remove_outliers(data, columns):
    # Use the IQR method to remove outliers for each column
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter out the outliers
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

def make_inference(model, input_data):
    pred = model.predict(input_data)
    return pred

app = Flask(__name__)
@app.route('/', methods=['GET'])
def home_route():
    return render_template('home1.html')

@app.route('/add', methods=['POST'])
def add_numbers():
    user_data = {}
    try:
        # Collect user input from the form and convert to float
        user_data['CurrentPopulation'] = float(request.form['cp'])
        user_data['PopulationDensity'] = float(request.form['pd'])
        user_data['BirthRate'] = float(request.form['br'])
        user_data['DeathRate'] = float(request.form['dr'])
        user_data['ImmigrationRate'] = float(request.form['ir'])
        user_data['UnemploymentRate'] = float(request.form['ur'])
        user_data['AverageIncome'] = float(request.form['ai'])
        user_data['HealthcareAccess'] = float(request.form['ha'])
        user_data['CrimeRate'] = float(request.form['cr'])
        
        # Education level encoding
        el = request.form['el']
        user_data['EducationLevel_High School'] = 1 if el == '1' else 0
        user_data['EducationLevel_Master'] = 1 if el == '4' else 0
        user_data['EducationLevel_PhD'] = 1 if el == '3' else 0
        
        user_df = pd.DataFrame([user_data])
        
        # Define numerical columns and apply the scaler (use transform, not fit_transform)
        numerical_cols = [
            'CurrentPopulation', 'PopulationDensity', 'BirthRate', 'DeathRate', 
            'ImmigrationRate', 'UnemploymentRate', 'AverageIncome', 
            'HealthcareAccess', 'CrimeRate'
        ]
        user_df[numerical_cols] = standard.transform(user_df[numerical_cols])
        
        # Select only the important features
        user_df_imp = user_df[imp_feats]
        
        # Make inference using the model
        result = make_inference(model_new, user_df_imp)
        result=''+str(result[0])
    except Exception as err:
        print(f"Error: {err}")
        result = 'Error'
    return render_template('result.html', result=result)

    
if __name__ == '__main__':
    filepath="population_growth_prediction_dataset.csv"
    x_train, x_test, y_train, y_test, standard = load_dataset(filepath)
    ver="v3.0"
    model=train(x_train, y_train, ver)
    x_train_imp,imp_feats=feature_importance(model,x_train)
    imp_feats_model = train(x_train_imp, y_train, ver + "_imp")
    x_test_imp=x_test[imp_feats]
    eval(imp_feats_model, x_test_imp, y_test)
    model_new=load_model("v3.0_imp")
    app.run(debug=True)

    





