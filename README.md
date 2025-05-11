# Weather Forecasting Using Stacked GRU RNN

This project implements a time series forecasting model for weather data using a stacked GRU-based Recurrent Neural Network (RNN) in Keras. The model predicts future temperature values based on historical weather features from the Jena Climate dataset. The code is provided as a Jupyter Notebook, compatible with Google Colab.

## Features
- Loads and preprocesses the [Jena Climate dataset](https://www.kaggle.com/datasets/pankrzysiu/weather-archive-jena)
- Normalizes and prepares time series data for supervised learning
- Uses a custom Python generator to create input/output samples for training, validation, and testing
- Builds a stacked GRU neural network with dropout regularization
- Trains and evaluates the model using Mean Absolute Error (MAE)
- Visualizes training and validation loss

## Technologies Used
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib

## How to Run
1. Download the [Jena Climate dataset (jena_climate_2009_2016.csv)](https://www.kaggle.com/datasets/pankrzysiu/weather-archive-jena) and place it in the notebook's working directory.
2. Open `Weather_forcasting_using_GRU_RNN.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter Notebook.
3. Run all cells in order.
4. (Optional) If running locally, first install the dependencies listed in `requirements.txt`:

    ```
    pip install -r requirements.txt
    ```

## Results
- The trained GRU model predicts future temperature values with low mean absolute error.
- Training and validation loss plots are included in the notebook.

## Dataset
- [Jena Climate Dataset](https://www.kaggle.com/datasets/pankrzysiu/weather-archive-jena)

## Author
- [Amirfarhad](https://github.com/Rubick666)
