from catboost import CatBoostClassifier  # Import the correct library
import pandas as pd
# Read CSV files from data folder
X_train = pd.read_csv('data/x_train.csv')
y_train = pd.read_csv('data/y_train.csv')


model = CatBoostClassifier(auto_class_weights='Balanced')
# training model on train set
model.fit(X_train, y_train)