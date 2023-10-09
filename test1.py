import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(0)
data = {
    'Magnitude': np.random.uniform(3.0, 9.0, 1000),
    'Depth': np.random.uniform(0.0, 700.0, 1000),
    'Distance_to_Fault': np.random.uniform(0.0, 100.0, 1000),
    'Time_Since_Last_Earthquake': np.random.uniform(0.0, 365.0, 1000),
    'Earthquake_Label': np.random.randint(2, size=1000)
}

df = pd.DataFrame(data)

X = df.drop('Earthquake_Label', axis=1)
y = df['Earthquake_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

