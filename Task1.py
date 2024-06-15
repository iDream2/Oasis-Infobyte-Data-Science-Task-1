import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

#Saving the model using the pickle library ~ 
import pickle

#Creating UI for the model 
import streamlit as st

df = pd.read_csv("./Iris.csv")
print(df.head())

le = LabelEncoder()
le.fit(df["Species"])
df["Species"] = le.transform(df["Species"])
y = df["Species"]
X = df.drop(["Id", "Species"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

neigh = KNeighborsClassifier(n_neighbors=3 )
neigh.fit(X_train, y_train)

#dumping the model ~ 
pickle.dump(neigh, open('neigh.pkl', 'wb'))


pred = neigh.predict(X_test)
print("\nAll the prediction made based on the X_test\n")
print(pred)


accuracy = accuracy_score(y_test, pred)
print("Accuracy Score : ", accuracy )

print(neigh.score(X_test, y_test))

loaded_model = pickle.load(open('neigh.pkl', 'rb'))
prediction = loaded_model.predict([[5.4,3.9,1.7,0.4]])
species = le.inverse_transform(prediction)
print(f"Predicted Species: {species[0]}")


#THE UI ~ 
st.write("Task-1")
SepalLengthCm = st.number_input("SepalLengthCm")
SepalWidthCm = st.number_input("SepalWidthCm")
PetalLengthCm = st.number_input("PetalLengthCm")
PetalWidthCm = st.number_input("PetalWidthCm")




if st.button("Predict"):
    loaded_model = pickle.load(open('neigh.pkl', 'rb'))
    output = loaded_model.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])
    species = le.inverse_transform(output)
    st.write(f"Predicted Species: {species[0]}")

