import streamlit as st
import pickle
import pandas as pd

# 1. Load the pre-trained model
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# 2. Set the title of your Streamlit application
st.title('Iris Species Prediction App')
st.write('Enter the measurements of the Iris flower to predict its species.')

# 3. Create input fields for each of the four Iris features
sepal_length = st.slider('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.slider('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = st.slider('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0, step=0.1)
petal_width = st.slider('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# 4. Create a 'Predict' button
if st.button('Predict'):
    # a. Create a Pandas DataFrame from the user's input values
    input_data = pd.DataFrame([{
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }])
    
    # b. Use the loaded model to make a prediction
    prediction = model.predict(input_data)
    
    # c. Display the predicted Iris species to the user
    st.success(f'The predicted Iris species is: {prediction[0]}')
