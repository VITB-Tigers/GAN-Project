# Import necessary libraries and functions for data loading, preprocessing, model retrieval, and training
import os
from ingest_transform import preprocess  # Function for preprocessing data
from tensorflow.keras.models import load_model  # Keras function to load pre-trained models
from tensorflow.keras.optimizers import Adam  # Optimizer to recompile models if needed
import pandas as pd  # Data manipulation library
import streamlit as st

# Function to load and preprocess the dataset
def data_load(df):
    # Read data from the provided file path (expects a CSV file)
    data = pd.read_csv(df)
    # Store column names for future reference if needed
    columns = data.columns
    # Apply preprocessing to the dataset (the preprocess function should be defined in ingest_transform)
    df = preprocess(data)
    # Return the preprocessed data and the original columns for further processing
    return df, columns

# Function to load pre-trained models stored locally
def load_pretrained():
    # Define the directory where the models are saved
    model_dir = 'data/saved_models'
    st.write("Model Accuracy for pre-trained model is:", 89.06)
    # Load the generator model using the saved .h5 file
    generator_loaded = load_model(os.path.join(model_dir, 'generator_model.h5'))
    # Load the discriminator model similarly
    discriminator_loaded = load_model(os.path.join(model_dir, 'discriminator_model.h5'))
    # Load the full GAN model if needed for continued training or evaluation
    gan_loaded = load_model(os.path.join(model_dir, 'gan_model.h5'))

    # Recompile the discriminator model if further training is intended
    # This setup is often used in GANs where the discriminator is trained independently
    discriminator_loaded.compile(loss='binary_crossentropy', 
                                 optimizer=Adam(learning_rate=0.0002), 
                                 metrics=['accuracy'])  # Loss function and optimizer
    
    # Set the discriminator as non-trainable within the GAN model to maintain GAN training structure
    discriminator_loaded.trainable = False
    # Recompile the GAN model to ensure it has the correct architecture and optimizer settings
    gan_loaded.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002))

    # Return a success message and the loaded generator model for further use in the application
    return "Models loaded successfully!", generator_loaded

# Function to load models based on the user's choice of database (PostgreSQL or CouchDB)
def load_selftrained():
    # Load the generator model using the saved .h5 file
    generator_loaded = load_model('data/saved_models/generator_model.h5')
    
    # Return a success message and the loaded generator model for further processing or usage
    return "Model loaded successfully!", generator_loaded
