import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")
st.header('Upload file to predict Iris flower')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
     
    # Can be used wherever a "file-like" object is accepted:
   dataframe = pd.read_csv(uploaded_file)
   st.write(dataframe)

st.sidebar.header('User Input Parameters')
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features
    
 
    
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

data = pd.read_csv('https://raw.githubusercontent.com/FatinDhaniel/my-final-assignment-AAA/main/IRIS.csv', sep=',',skipinitialspace=True)


X=data.drop('species',axis=1)
y=data['species']



clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)



st.subheader('Class labels and their corresponding index number')
list = ['setosa','versicolor','virginica']
ser =pd.Series(list)
st.write(ser)

st.subheader('Prediction')
#st.write(ser[prediction])
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)




