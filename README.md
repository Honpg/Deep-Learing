# # Time Series Forecasting for Deep Learning Project

Welcome to the Deep Learning project repository! This project explores various deep learning techniques and their applications.

## Table of Contents

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Visualization](#visualization)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Introduction

The ability to predict water levels is crucial for disaster management, prevention, and mitigating the impact of floods. While physically-based models often yield good results, developing forecasting systems with these models demands significant computational resources and extensive hydrogeomorphological data. In contrast, data-driven forecasting models, particularly those using machine learning (ML) techniques, are generally simpler and quicker to develop. In recent decades, statistical ML techniques have made considerable strides in creating high-performance, cost-effective data-driven forecasting systems.

Deep learning models, in particular, offer fast and reliable forecasting due to their ability to extract data with high specificity, a critical factor for optimal model learning. Among these, the Long Short-Term Memory (LSTM) model is renowned for its effectiveness in handling long-term univariate time series data, enabling data memorization through its neural network structure. LSTM, an advanced form of Recurrent Neural Network (RNN), processes data through multiple input ports, extracting essential features and removing noise, thereby enhancing the model's performance.

In this work, we review and evaluate both individual methods and hybrid approaches that combine various deep learning models, including Convolutional Neural Networks (CNNs), Transformer Networks, and LSTMs. Additionally, we incorporate Attention Mechanisms (AM) to further refine the focus of the models on relevant parts of the data, improving the ability to capture complex relationships.

The proposed techniques for water level forecasting encompass LSTM models, hybrid models combining CNN, Transformer, LSTM, and AM, and other deep learning statistical models. The performance of these techniques was assessed using hourly data collected from 2008 to April 2015 at the Hung Yen hydrological station. The results from testing on these extensive datasets offer the most specific and accurate evaluations of the models we selected.

## Data Preprocessing
Datasets (Vu Quang, Hanoi, and Hung Yen).

Data preprocessing is a crucial step in preparing the dataset for model training. The following steps were undertaken:

- **Handling Missing Values**: Erroneous entries in the `Hour` column were replaced to ensure data consistency.
- **Date and Time Conversion**: The `Date` and `Hour` columns were converted into a unified `DateTime` index for easier manipulation and analysis.
- **Column Removal**: Unnecessary columns were dropped post-conversion to streamline the dataset.

## Model Training and Evaluation

This section contains implementations of various neural network architectures, including:

- **Convolutional Neural Networks (CNN1D)**
- **Long Short-Term Memory Networks (LSTM)**
- **Transformer Networks(AM)**
- **Hybrid models combining different architectures**

**Model Definition**: Each model is implemented and tested to determine its effectiveness in forecasting water levels followed by a Dropout layer to prevent overfitting.

**Training**: The model is compiled using the Mean Squared Error (MSE) loss function and the Adam optimizer. Training is conducted with early stopping based on validation loss to avoid overfitting.

**Evaluation**: Performance evaluation metrics including MAE, RMSE, R score, FSD, and NSE, providing a quantitative measure of the model's accuracy.

## Visualization

Visualizations play a critical role in understanding the model's performance and data trends:

- **Data Visualization**: Initial data plots showcase water level trends over time, helping identify patterns and anomalies.
- **Prediction Visualization**: Comparative plots illustrate actual versus predicted water levels, as well as forecasts for various time horizons, demonstrating the model's predictive capability.

## Installation

To get started with the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Honpg/Deep-Learing.git
cd Deep-Learing
pip install -r requirements.txt
```

## Usage
### Training a Model
To train a model, use the following command:

```bash
python train.py --config config.h5
```

**or**
```bash
python train.py --config config.keras
```

### Evaluating a Model
To evaluate a model, run:
```bash
python evaluate.py --model path/to/model --data path/to/data
```

### Visualization
To visualize the training process and results, use:
```bash
python visualize.py --logdir logs/
```

## Contributing

We welcome contributions to this project! Please fork the repository and create a pull request with your changes. For major changes, please open an issue to discuss what you would like to change:

**1.Fork the repository.**
**2.Create a new branch for your feature or bug fix.**
**3.Make your changes and commit them with clear messages.**
**4.Push your changes to your forked repository.**
**5.Submit a pull request with a description of your changes.**

## Acknowledgments
Special thanks to all contributors and the open-source community for their valuable resources and tools.

**Mentor** : [Prof.Hong Phan Thi Thu](https://scholar.google.com/citations?user=yXkziQIAAAAJ&hl=en&oi=ao)

**Member** : Khoi Nguyen Ta,Van Quoc Hoan Doan




