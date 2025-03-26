# Mock Data Generator - A GAN Project

## Table of Contents
- [Project Description](#project-description---fake-data-generation-for-churn-prediction)
- [Data Definition](#data-definition)
- [Directory Structure](#directory-structure)
- [Program Flow](#program-flow)
- [Steps to Run](#steps-to-run)

---

## Project Description - Mock Data Generation for Churn Prediction

A telecom company wants to simulate customer churn data to better understand patterns that may lead to customer attrition. Using synthetic data, the company can experiment with different churn scenarios, test predictive models, and create targeted strategies to reduce churn without compromising actual customer data privacy.

### Industry

Telecommunications

### Problem Statement

Creating realistic data for customer churn analysis is challenging, especially when privacy regulations restrict access to sensitive customer information. By generating synthetic churn data, the company can develop and test predictive models to understand how different factors influence churn, allowing for actionable insights without using actual customer records.

### Objective

The objective is to develop a process for generating realistic synthetic churn data that resembles real-world data in key characteristics, such as:

* **Demographics** (e.g., age, gender)
* **Service Subscriptions** (e.g., internet, phone, streaming services)
* **Billing and Contract Information** (e.g., monthly charges, contract type)

This synthetic data will allow for training and validating predictive models to identify patterns that lead to churn, enabling the telecom company to strategize on customer retention efforts effectively.

---

## Data Definition

The synthetic dataset contains features that mirror real-world customer data, including:

* **Customer demographics** (e.g., age, gender, region)
* **Service subscription details** (e.g., internet, phone, streaming services)
* **Financial data** (e.g., monthly charges, contract type, total charges)
* **Tenure** (e.g., length of subscription)
* **Churn status** (target variable indicating whether a customer has churned)

---

### Directory Structure
```
├── code\                              # Contains the source code files for the project.
│   ├── app.py                         # Main application script.
│   ├── generate.py                    # Script for generating data or models.
│   ├── ingest_transform.py            # Script for data ingestion and transformation.
│   ├── load.py                        # Script for training the model.
│   └── train.py                       # Script for training the model.
├── data\                              # Contains data files and model files.
│    └── master\                       # Directory containing the main dataset.
│        └── telecom_customer_data.csv # CSV file with customer data for analysis.
│    └── pretrained_models\            # Contains pretrained model files.
│        ├── discriminator_model.h5    # Pretrained discriminator model.
│        ├── gan_model.h5              # Pretrained GAN model.
│        └── generator_model.h5        # Pretrained generator model.
│    └── saved_models\                 # Contains saved model files after training.
│        ├── discriminator_model.h5    # Saved discriminator model.
│        ├── gan_model.h5              # Saved GAN model.
│        └── generator_model.h5        # Saved generator model.
├── .gitignore                         # Git ignore file for excluding files from version control.
├── README.md                          # Documentation describing the project purpose, usage, and structure.
└── requirements.txt                   # Dependencies required for the project
```

---

## Program Flow

1. **Data Generation**: Generate synthetic churn data using a generative model (e.g., GANs, Variational Autoencoders) that mirrors real-world characteristics. [data\_generation.py]
2. **Data Ingestion**: Load synthetic data into a database (MongoDB or SQLite) for further processing and analysis. [ingest\_transform\_mongodb.py, ingest\_transform.py]
3. **Data Transformation**: Preprocess the synthetic data, including encoding categorical features, handling missing values, and normalizing numerical values. The data is then split into training, testing, and validation sets. [ingest\_transform.py]
4. **Model Training**: Train deep learning models (e.g., using TensorFlow or PyTorch) to understand patterns in the synthetic data, with cross-validation and hyperparameter tuning. [train.py]
5. **Manual Prediction**: Enable manual input of synthetic customer data for churn prediction. [generate.py]
6. **Web Application**: A Streamlit app that allows interaction with the data generation pipeline and model, providing churn predictions based on synthetic data input. [app.py]

> *Note*: The data generated also includes CustomerID. Therefore duplicate CustomerID might be possible. Please Drop the CustomerID before using for data analysis.

---

## Steps to Run

1. Clone the repository:
    ```
    git clone https://github.com/VITB-Tigers/GAN-Project
    ```

2. Create and activate a virtual environment (recommended). If using Conda:
    ```
    conda create -n env_name python==3.11.5 -y
    conda activate env_name
    ```
    > *This project uses Python 3.11.5*

3. Install the necessary packages:
    ```
    pip install -r requirements.txt
    ```

4. Launch Application: Run the following command in your terminal/command prompt
   ```
   streamlit run code/app.py
   ```

5. Model Configuration:
   - In "Model Config" tab, enter path where your training data is located
   - Example path: `data/master`
   - Click "Use this Data Path" to confirm

6. Model Training:
   - Go to "Model Training" tab
   - Set number of epochs (recommended: start with 100)
   - Click "Train GAN Model" button
   - Wait for training to complete (progress shown in status bar)

7. Generate Synthetic Data:
   - Switch to "Data Generation" tab
   - Choose number of samples to generate
   - Select model type:
     - "Pretrained GAN" for quick results
     - "Self-trained GAN" to use your trained model
   - Click "Generate Fake Data"
   - Once generated, use "Download Data" to save as CSV

> Note: Ensure your training data is in CSV format and follows the same structure as the example telecom customer data.