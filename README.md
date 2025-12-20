# Credit Card Fraud with Neural Networks

This repo demonstrates all data science aspects of working with the Credit Card fraud dataset (CITE ME). Specifically, it looks into the aspects of tuning a neural network for increased performance with the most minimal feature engineering possible.

# Project Structure

- `data/`
  - Contains base data as well as pkl files for prepped and scaled test/train data

- `notebooks/`
  - Handles a small collections of notebooks as necessary. These are not end to end notebooks, simply small tests and manipulations.

- `plots/`
  - Contains all of the saved charts and images generated throughout the project.

- `src/`
  - All of the good stuff in the project.

  - `src/explore.py`
    - Contains the EDA of the data. Built as a script that can be run in cells in a REPL.

  - `src/preprocess.py`
    - Preprocessing of the data for the training. Reads in the base data, creates a test/train split, and scales the data. Saves the scaler and the X_train_scaled, X_test_scaled, y_train, and y_test data as `.pkl` files.

  - `src/record.py`
    - Contains a single Recorder class for recording results.

  - `srcs/train.py`
    - The training script for the NN.

  - `src/models`
    - Contains any of the model (scalers, neural networks) files
