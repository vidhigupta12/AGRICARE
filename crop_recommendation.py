import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

acc = []
model = []
df = pd.read_csv('data.csv')
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    features, target, test_size=0.2, random_state=2)

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain, Ytrain)

filename = "final_model.sav"
joblib.dump(RF, filename)
