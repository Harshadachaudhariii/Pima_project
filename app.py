import os
import pandas as pd 
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# file paths

MODEL_PATH = 'model.pkl'
REAL_TIME_PREDICTIONS_PATH = 'data/real_time_predictions.csv'
BATCH_PREDICTIONS_PATH = 'data/batch_predictions.csv'
ONLINE_DATA_PATH = 'data/online_data.csv'

REQUIRED_FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

def fetch_and_save_data():
    '''Fetch the dataset from an online API'''
    url = f'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    columns = REQUIRED_FEATURES + ['Outcome']
    data = pd.read_csv(url,header = None, names = columns)
    os.makedirs('data', exist_ok=True)
    data.to_csv(ONLINE_DATA_PATH, index= False)
    print("Dataset downloaded and saved into data folder.")
    return data

def train_and_save_model():
    '''Training the model and save it to a file'''
    if not os.path.exists(ONLINE_DATA_PATH):
        data = fetch_and_save_data()
    else:
        data = pd.read_csv(ONLINE_DATA_PATH)
    X = data.drop(columns = ['Outcome'])
    Y = data['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)    
    model.fit(X_train, y_train)
    
    # evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model trained with accuracy: {accuracy*100:.4f}%')
    
    #save the model
    joblib.dump(model, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')
    
def load_model():
    '''Load the trained model from file'''
    if not os.path.exists(MODEL_PATH):
        print('Model not found Training an new model...')
        train_and_save_model()
        
    return joblib.load(MODEL_PATH)

model= load_model()

def validate_input(data, required_features):
    '''validate input data for missing features'''
    missing_features = [feature for feature in required_features if feature not in data]
    if missing_features:
        raise ValueError(f"Missing feature:{','.join(missing_features)}")
        
