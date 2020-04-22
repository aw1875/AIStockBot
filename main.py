# Imports
import quandl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# Get stock data
quandl.ApiConfig.api_key = "DK7VgAuQs_k8zE6fiLqc"
df = quandl.get("EOD/AAPL")

# Get Adj. Close
df = df[['Adj. Close']]

# Prediction variable - 1 is the number of days
forecast_out = 30

# Create target column and shift forcecast_out units up
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

# Create independent data set X
X = np.array(df.drop(['Prediction'], 1))    # Convert df to array
X = X[:-forecast_out]   # Remove last n days where n is forecast_out

# Create dependent data set Y
Y = np.array(df['Prediction'])    # Convert df to array
Y = Y[:-forecast_out]   # Remove last n days where n is forecast_out

# Split data into training (80%) and testing (20%)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Create and train SVM (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

# Test SVM model
svm_confidence = svr_rbf.score(x_test, y_test)
# print("svm confidence: ", svm_confidence)

# Create and train LRM
lrm = LinearRegression()
lrm.fit(x_train, y_train)

# Test LRM model
lrm_confidence = lrm.score(x_test, y_test)
# print("lrm confidence: ", lrm_confidence)

# Set x_forecast to last 30 days from Adj. Close
x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
# print(x_forecast)

# Print next n days where n is forecase_out
lrm_prediction = lrm.predict(x_forecast)
svr_prediction = svr_rbf.predict(x_forecast)