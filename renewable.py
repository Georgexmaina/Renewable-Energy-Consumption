import streamlit as st
import pandas as pd

st.title("Renewable Energy Consumption App")

ren_ernergy = pd.read_csv('C:\\Users\\maina\\OneDrive\\Desktop\\Renewable energy\\WorldBank Renewable Energy Consumption_WorldBank Renewable Energy Consumption.csv')
st.write(ren_ernergy.head(5))

#having the user input to check number of rows
num_rows = st.slider("select the number of rows", min_value =1,max_value=len(ren_ernergy), value=5)
st.write("Here are the rows you have selected ")
st.write(ren_ernergy.head(num_rows))

#Shape of the dataset
if st.checkbox("Number of rows and colums in the data(data shape)"):
    st.write(ren_ernergy.shape)

 #checkbox for duplicates
if st.checkbox ("Check total number of duplicates in the dataset"):
 st.write(ren_ernergy.duplicated().sum())



############MACHINE LEARNING##############

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Encoding country name column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
encoded_columns= ['Country Code', 'Country Name', 'Income Group', 'Indicator Code', 'Indicator Name', 'Region', 'Year']
le_dict = {col: LabelEncoder() for col in encoded_columns}
for column in encoded_columns:
  le_dict[column].fit(ren_ernergy[column])
  ren_ernergy[column] = le_dict[column].transform(ren_ernergy[column])

#Encode target variable 
le_target = LabelEncoder()
ren_ernergy['Income Group'] = le_target.fit_transform(ren_ernergy['Income Group'])

#Give x and y variables
x = ren_ernergy.drop(['Income Group'], axis=1)
y = ren_ernergy['Income Group']
#train and split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#model variable
model = RandomForestClassifier()
#fit the model
model.fit(x_train, y_train)
#predict the model
y_pred = model.predict(x_test)
#Get accuracy
accuracy = model.score(x_test, y_test)
print(accuracy)

if st.checkbox('Model Accuracy'):
    st.write(accuracy)

# User input for new data
st.sidebar.write("## Enter new data for prediction")
Country_Code = st.sidebar.selectbox("Country Code", le_dict['Country Code'].classes_)
Country_Name = st.sidebar.selectbox("Country Name", le_dict['Country Name'].classes_)
Income_Group = st.sidebar.selectbox("Income Group", le_dict['Income Group'].classes_)
Indicator_Code = st.sidebar.selectbox("Indicator Code", le_dict['Indicator Code'].classes_)
Indicator_Name = st.sidebar.selectbox("Indicator Name", le_dict['Indicator Name'].classes_)
Region = st.sidebar.selectbox("Region", le_dict['Region'].classes_)

Energy_Consump = st.sidebar.number_input("Energy Consump.")

# Encode user input
encoded_input = [
    le_dict['Country Code'].transform([Country_Code])[0],
    le_dict['Country Name'].transform([Country_Name])[0],
    le_dict['Income Group'].transform([Income_Group])[0],
    le_dict['Indicator Code'].transform([Indicator_Code])[0],
    le_dict['Indicator Name'].transform([Indicator_Name])[0],
    le_dict['Region'].transform([Region])[0],
    Energy_Consump
]

income_group_map = {
    0: "High income",
    1: "Low income",
    2: "Lower middle income",
    3: "Upper middle income"
}


# Predict using the model
if st.sidebar.button('Income Group'):
    prediction = model.predict([encoded_input])[0]
    predicted_income_group = income_group_map[prediction]
    st.sidebar.write('Income Group', predicted_income_group)