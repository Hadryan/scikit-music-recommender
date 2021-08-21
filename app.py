import pandas as pd

music_data = pd.read_csv('music.csv')

shape = music_data.shape
print(music_data, shape)


X = music_data.drop(columns=['genre'])  # Input set
Y = music_data['genre']                 # Output set

# Predict

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(X, Y)
predictions = model.predict([ [21, 1], [22, 0] ])

print(predictions)

# Train & Test

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

new_model = DecisionTreeClassifier()

new_model.fit(X_train, Y_train)
predictions = new_model.predict(X_test)

score = accuracy_score(Y_test, predictions)
print(score)


# Saving trained data

import joblib

filename = 'music-recommender.joblib'
joblib.dump(new_model, filename)

# Load trained data

saved_model = joblib.load(filename)

predictions = saved_model.predict([ [21, 1] ])
print(predictions)


# Visualizing decision tree

from sklearn.tree import export_graphviz

export_graphviz(model, out_file='music-recommender.dot',
               feature_names=['age', 'gender'], class_names=sorted(Y.unique()),
               label='all', rounded=True, filled=True)