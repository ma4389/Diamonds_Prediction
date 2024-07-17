import streamlit as st
import io
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Function to load data

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Directly read the CSV file
file_path = "diamonds.csv"
dim = load_data(file_path)
st.write("Data Overview:")
st.dataframe(dim.head(5))

# Drop unnecessary column
if 'Unnamed: 0' in dim.columns:
    dim.drop('Unnamed: 0', axis=1, inplace=True)

# Display data info
st.write("Data Information:")
buffer = io.StringIO()
dim.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Show columns
st.write("Columns in the dataset:", dim.columns.tolist())

# Check for missing values
st.write("Missing values in each column:", dim.isna().sum())

# Check for duplicates
st.write("Number of duplicated rows:", dim.duplicated().sum())

# Drop duplicates
dim.drop_duplicates(inplace=True)
st.write("Number of duplicated rows after dropping duplicates:", dim.duplicated().sum())

# Data shape
st.write("Data shape:", dim.shape)

# Describe data
st.write("Data Description:")
st.write(dim.describe())

# Data types
st.write("Data Types:", dim.dtypes)

# Unique values in 'cut' column
if 'cut' in dim.columns:
    st.write("Unique values in 'cut' column:", dim['cut'].unique())

# Number of unique values in each column
st.write("Number of unique values in each column:", dim.nunique())


# Heatmap of correlations
st.write("Correlation Heatmap:")
nu_cols = dim.select_dtypes(exclude='object')
plt.figure(figsize=(10, 6))
sns.heatmap(nu_cols.corr(), annot=True)
st.pyplot()
# Split data into features and target
x = dim.drop('price', axis=1)
y = dim['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Preprocessor
cat_cols = dim.select_dtypes(include='object')
num_cols = dim.select_dtypes(exclude='object').drop('price', axis=1)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols.columns),
    ('num', StandardScaler(), num_cols.columns)
])

# Linear Regression Model
st.write("Linear Regression Model:")
pipe_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

pipe_lr.fit(x_train, y_train)
lr_score = pipe_lr.score(x_test, y_test)
st.write(f"Linear Regression Model Accuracy: {lr_score}")

# K-Neighbors Regressor Model
st.write("K-Neighbors Regressor Model:")
pipe_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('model', KNeighborsRegressor(n_neighbors=10))
])

pipe_knn.fit(x_train, y_train)
knn_score = pipe_knn.score(x_test, y_test)
st.write(f"K-Neighbors Regressor Model Accuracy: {knn_score}")

# Random Forest Regressor Model
st.write("Random Forest Regressor Model:")
pipe_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

pipe_rf.fit(x_train, y_train)
rf_score = pipe_rf.score(x_test, y_test)
st.write(f"Random Forest Regressor Model Accuracy: {rf_score}")

# Display the best model
best_model_name = max([("Linear Regression", lr_score), 
                       ("K-Neighbors Regressor", knn_score), 
                       ("Random Forest Regressor", rf_score)], key=lambda x: x[1])[0]
st.write(f"The best model is: {best_model_name} with accuracy of {max(lr_score, knn_score, rf_score)}")
