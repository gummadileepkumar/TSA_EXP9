# EX.NO.09        A project on Time series analysis on National Stock Exchange using ARIMA model 

### Date: 
### Developed by: Gumma Dileep Kumar
### Register number: 212222240032

### AIM:
To Create a project on Time series analysis on National Stock Exchange using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of National Stock Exchange. 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:


#### Import the neccessary packages

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
```

#### Load the dataset
```
data = pd.read_csv("/content/infy_stock1.csv")
```
#### Convert 'Date' column to datetime format
```
data['Date'] = pd.to_datetime(data['Date'])
```
#### Set 'Date' column as index
```
data.set_index('Date', inplace=True)
```
#### Arima Model
```
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=len(test_data))

    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()

    print("Root Mean Squared Error (RMSE):", rmse)

arima_model(data, 'Volume', order=(5,1,0))

```




### OUTPUT:

![Screenshot 2024-11-06 092140](https://github.com/user-attachments/assets/1eb0df5b-b6b8-495f-8fbe-9cda1539c390)




### RESULT:
Thus the project on Time series analysis on National Stock Exchange based on the ARIMA model using python is executed successfully.
