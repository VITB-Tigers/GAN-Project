# Import necessary libraries and modules
import streamlit as st  # Streamlit library for web applications
from sklearn.preprocessing import MinMaxScaler  # Scaler to normalize data values

# Define mappings to convert categorical values to numerical representations for model use
contract_mapping = {'One year': 0, 'Two year': 1, 'Month-to-month': 2}
internet_service_mapping = {'DSL': 0, 'Fiber optic': 1}
payment_mapping = {'Mailed check': 0, 'Bank transfer': 1, 'Credit card': 2, 'Electronic check': 3}
agree_mapping = {'Yes': 0, 'No': 1}

# Initialize a scaler instance for data normalization
scaler = MinMaxScaler()

# Function to label categorical data with numeric values for model compatibility
def labelling(data):
    """
    Convert categorical features to numerical values using predefined mappings.

    Parameters:
    - data (DataFrame): Input data with categorical features.

    Returns:
    - data (DataFrame): Data with categorical features converted to numerical values.
    """
    global contract_mapping, internet_service_mapping, payment_mapping, agree_mapping

    # Apply mappings to the respective columns
    data['Contract'] = data['Contract'].map(contract_mapping)
    data['InternetService'] = data['InternetService'].map(internet_service_mapping)
    data['PaymentMethod'] = data['PaymentMethod'].map(payment_mapping)
    data['OnlineSecurity'] = data['OnlineSecurity'].map(agree_mapping)
    data['TechSupport'] = data['TechSupport'].map(agree_mapping)
    data['StreamingTV'] = data['StreamingTV'].map(agree_mapping)
    data['StreamingMovies'] = data['StreamingMovies'].map(agree_mapping)

    # Fill any missing values in 'InternetService' with the mean value of the column
    data['InternetService'] = data['InternetService'].fillna(data['InternetService'].mean())

    return data

# Function to revert numerical labels to their original categorical values
def delabel(df):
    """
    Convert numeric labels back to their original categorical values.

    Parameters:
    - df (DataFrame): DataFrame with numerical labels.

    Returns:
    - df (DataFrame): Data with categorical features restored to their original string values.
    """
    # Create reverse mappings for converting numbers back to category labels
    contract_reverse_mapping = {v: k for k, v in contract_mapping.items()}
    internet_service_reverse_mapping = {v: k for k, v in internet_service_mapping.items()}
    payment_reverse_mapping = {v: k for k, v in payment_mapping.items()}
    agree_reverse_mapping = {v: k for k, v in agree_mapping.items()}
    
    # Apply reverse mappings to convert numeric columns back to categorical format
    df['Contract'] = df['Contract'].round().astype('int').map(contract_reverse_mapping)
    df['InternetService'] = df['InternetService'].round().astype('int').map(internet_service_reverse_mapping)
    df['PaymentMethod'] = df['PaymentMethod'].round().astype('int').map(payment_reverse_mapping)
    df['OnlineSecurity'] = df['OnlineSecurity'].round().astype('int').map(agree_reverse_mapping)
    df['TechSupport'] = df['TechSupport'].round().astype('int').map(agree_reverse_mapping)
    df['StreamingTV'] = df['StreamingTV'].round().astype('int').map(agree_reverse_mapping)
    df['StreamingMovies'] = df['StreamingMovies'].round().astype('int').map(agree_reverse_mapping)

    # Convert other necessary columns to integers for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].round().astype('bool')
    df['PaperlessBilling'] = df['PaperlessBilling'].round().astype('bool')
    df['Churn'] = df['Churn'].round().astype('bool')
    df['CustomerID'] = df['CustomerID'].round().astype('int')
    df['Tenure'] = df['Tenure'].round().astype('int')
    
    return df

# Function to preprocess data by applying labelling and scaling
def preprocess(data):
    """
    Label categorical data and scale numerical values for model compatibility.

    Parameters:
    - data (DataFrame): Input DataFrame to be processed.

    Returns:
    - scaled_data (array): Processed and scaled data ready for model input.
    """
    data = labelling(data)  # Apply labelling function
    scaled_data = scaler.fit_transform(data)  # Scale the data
    return scaled_data  # Return scaled data for model input

