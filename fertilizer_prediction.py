import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from bs4 import BeautifulSoup

df = pd.read_csv('fertilizer.csv')


X=df.drop(['class'],axis=1)
Y=df['class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

cl = RandomForestClassifier()
cl.fit(X_train,Y_train)

filename = "fertilizer_model.sav"
joblib.dump(cl, filename)