# Spam-Checker
## Description

This project is a PyQt-based desktop application that checks text messages for spam. The classification is performed using a machine learning model, Multinomial Naive Bayes (MultinomialNB), which was trained in advance and then integrated into the application.

The main idea of the project is to demonstrate how machine learning models can be used in real desktop applications with a graphical user interface.

## Project Goals

1) Demonstrate skills in working with data and machine learning libraries such as pandas and scikit-learn, including text vectorization and Naive Bayes classification.

2) Create an application capable of classifying text messages as spam or not spam.

3) Apply machine learning techniques in the context of a desktop application built with PyQt6.

## Application Features

- Input of text messages for analysis.

- Display of the classification result (Spam / Not Spam).

## Libraries and Technologies

- Python

- PyQt6

- scikit-learn

- pickle

- pandas

## Dataset

- For this project, a dataset from the Kaggle platform was used. It contains text messages labeled as either spam or not spam.

- During preprocessing, several issues were identified:

- Some text fields contained commas, which caused the dataset to be incorrectly split into multiple columns.

- These columns were merged back into a single text column.

- The first column with class labels was converted into a binary format suitable for machine learning algorithms.

- This preprocessing was necessary to make the dataset consistent and usable for training the model.
## Machine Learning Model

- The following approach was used:

- A Multinomial Naive Bayes classifier was chosen as the model.

- The parameter alpha was set to 0.001 to make the classifier more sensitive to spam detection and improve recall.

- Text data was vectorized using TfidfVectorizer from the scikit-learn library.

- After training, both the model and the vectorizer were saved using the pickle module as binary pkl files.



- This design allows the application to work efficiently and independently of the training process.

