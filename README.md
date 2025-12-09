# Melbourne Temperature LSTM Forecast

Univariate LSTM model for forecasting daily minimum temperatures in Melbourne.  
The model takes the previous 30 days of temperatures and predicts the temperature for the next day.

## Dataset

- Source: *Daily Minimum Temperatures in Melbourne, Australia (1981–1990)* from Kaggle  
- Target: next-day minimum temperature  
- Features: only past temperature values

## Data Preparation

- Sorted by date and cleaned (invalid values → NaN → dropped)
- Temperatures scaled to \[0, 1\] with `MinMaxScaler`
- Sliding window approach:
  - Input sequence length: **30 days**
  - Each sample:  
    - `X`: temperatures at days *t-30 ... t-1*  
    - `y`: temperature at day *t*
- Train/test split: 80% / 20% by time

## Model Architecture

Implemented with TensorFlow/Keras:

- `LSTM(64, input_shape=(30, 1))`
- `Dense(1)` — regression output (next-day temperature)
- Loss: `mean_squared_error`
- Optimizer: `Adam`
- Training:
  - Epochs: 100
  - Batch size: 32
  - `validation_split=0.1` on the training set

## Results

Metrics are computed on the test set and reported in °C (after inverse scaling).

### RMSE Comparison

| Model                     | RMSE (°C) |
|---------------------------|-----------|
| Naive (tomorrow = today) | **2.48**  |
| 3-day moving average      | **2.62**  |
| LSTM (30-day window)      | **2.21**  |

The LSTM model outperforms both simple baselines, reducing the average prediction error to about **2.2°C**, which is reasonable given that it uses only historical temperature values without any additional weather features.
