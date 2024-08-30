# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

One kind of machine learning method that draws inspiration from the structure of the brain is the neural network regression model. It is highly proficient in recognizing intricate patterns in data and applying those patterns to forecast continuous numerical values.This involves dividing your data into training and testing sets, cleaning, and normalizing it. The model is trained on the training set, and its accuracy is assessed on the testing set. This entails selecting the quantity of neurons in each layer, the number of layers overall, and the kind of activation functions to be applied.The training data is fed into the model.The testing set is used to evaluate how effectively the model generalizes to fresh, untested data once it has been trained. Metrics like Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) are frequently used in this.Considering the

## Neural Network Model

![nn model](https://github.com/user-attachments/assets/2a9f2f1a-e17c-4747-8368-4844a8fc151f)

This layer has a single neuron (R¹ indicates that one-dimensional data is accepted). This implies that there is probably only one feature or predictor variable in your dataset.Add the schematic for the neural network model.R2 indicates that each layer has two-dimensional output, and there are two hidden layers, each containing two neurons. These layers analyze the incoming data and identify intricate patterns in it.Regression problems often include a single numerical result being predicted by the model, as evidenced by the final layer's single neuron.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: DHANASHREE M
### Register Number: 212221230018
```


import pandas as pd
import sklearn
import tensorflow
from tensorflow import keras
     

from sklearn.model_selection import train_test_split
     

from sklearn.preprocessing import MinMaxScaler
     

from tensorflow.keras.models import Sequential

     

from tensorflow.keras.layers import Dense
     

from google.colab import auth
import gspread
from google.auth import default
     

dataset=pd.read_csv('/content/EX1 - Sheet1.csv')
     

auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize(creds)
     

worksheet = gc.open('EX1').sheet1
data = worksheet.get_all_values()
     

dataset=pd.DataFrame(data[1:],columns=data[0])
dataset = dataset.astype({'x':'float'})
dataset = dataset.astype({'y':'float'})

dataset.head()

x=dataset[['x']].values
y=dataset[['y']].values
     

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
     

scaler=MinMaxScaler()
     

scaler.fit(x_train)


x_train1 = scaler.transform(x_train)
     

ai_brain = Sequential([Dense(8,activation='relu'),
                       Dense(10,activation='relu'),
                       Dense(1)])
     

ai_brain.compile(optimizer='rmsprop',loss ='mse')
ai_brain.fit(x_train,y_train,epochs=20)

loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
ai_brain.evaluate(x_test,y_test)
X_n1 = [[3]]

X_n1_1 = scaler.transform(X_n1)
ai_brain.predict(X_n1_1)  

```
## Dataset Information

![DATASET](https://github.com/user-attachments/assets/6485b292-1393-4427-ba61-e1caa0e66770)


## OUTPUT

### Training Loss Vs Iteration Plot

![PLOT](https://github.com/user-attachments/assets/e0ab0c49-d587-4df3-9a60-0e7a73e9cd4d)


### Test Data Root Mean Squared Error

![MSE](https://github.com/user-attachments/assets/eafd4e22-7c98-4bf2-bacc-06607c6fbf2e)


### New Sample Data Prediction

![DP](https://github.com/user-attachments/assets/46d79912-dd10-4271-a1a5-16427f5cd33b)


## RESULT

Thus a neural network regression model for the given dataset has been developed.
