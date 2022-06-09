import streamlit as st
import pandas as pd
#from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

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

#np.random.seed(151)  
#df2 = pd.DataFrame(np.random.randn(151),index = list(), columns = list('sepel_lenghth','sepel_width','petel lenghth','petel width','species'))
#data =['Sepel lenght']+['sepel width']+['petel length']+['petel width'] +['species']

X=data.drop('species',axis=1)
y=data['species']



#df2 = pd.DataFrame(np.random.randn(8, 4), index=datas, columns=['Sepel length', 'Sepel width', 'Petel lenght', 'Petel width','species'])

#X = df2.drop('species',axis=)
#Y= df('species')



#labelencoder = LabelEncoder()
#data['species']= labelencoder.fit_transform(data['species'])

#data.head()

#3X= features.drop('label',axis=1)

#y= features['data']

#X= pd.DataFrame(data['features'])
#y = data['species']


#X = iris.data
#Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

prediction = clf.predict(df)
#prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write['setosa','versicolor','virginica']

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
