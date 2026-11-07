# Iris Flower Classification

This project is a simple machine learning application that predicts the species of an iris flower from four measurements: sepal length, sepal width, petal length, and petal width. It uses a K-Nearest Neighbors (KNN) classifier with feature scaling and provides an interactive interface built with Streamlit.

## Features

- Predicts an iris species from user-provided flower measurements
- Trains a KNN model using the included Iris dataset
- Reports accuracy, a confusion matrix, and a classification report during training
- Saves the trained model as `iris_model.pkl` for use by the app

## Technologies

- Python
- Streamlit
- pandas and NumPy
- scikit-learn
- joblib

## Run the app locally

1. Install Python 3 and open a terminal in this project folder.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

Streamlit will show a local address for testing on your computer. This project does not currently have a publicly hosted demo, so the local address is not a public demo link.

## Train the model again

The pre-trained model is included. To train it again from the dataset and replace `iris_model.pkl`, run:

```bash
python train_model.py
```

The training script uses an 80/20 train-test split and a KNN classifier with three neighbors.

## Project files

- `app.py` — Streamlit interface for entering measurements and viewing predictions
- `train_model.py` — model training and evaluation script
- `explore.py` — basic dataset exploration
- `Iris.csv` — dataset used to train the model
- `iris_model.pkl` — saved trained model loaded by the app
- `requirements.txt` — Python dependencies
