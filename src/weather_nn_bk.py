from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(input_window_size, num_features)))
model.add(Dense(output_window_size * num_features))  # Adjust based on your specific output shape

model.compile(optimizer='adam', loss='mse')
model.fit(train_sequences, train_labels, epochs=50, batch_size=32, validation_split=0.2)


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Assume 'X' is your input features

# Regression labels
y_regression = weather_data[['temperature', 'humidity', 'wind_speed']]

# Classification labels
y_classification_rain = (weather_data['rain'] > 0).astype(int)  # 1 if rain, 0 if no rain
y_classification_sunny = (weather_data['sunshine'] > 0).astype(int)  # 1 if sunny, 0 if not sunny

# Input layer
input_layer = Input(shape=(input_feature_size,))

# Shared hidden layers
hidden_layer_1 = Dense(64, activation='relu')(input_layer)
hidden_layer_2 = Dense(32, activation='relu')(hidden_layer_1)

# Regression output layer
regression_output = Dense(3, name='regression_output')(hidden_layer_2)  # Assuming 3 regression targets

# Classification output layers
classification_output_rain = Dense(1, activation='sigmoid', name='classification_output_rain')(hidden_layer_2)
classification_output_sunny = Dense(1, activation='sigmoid', name='classification_output_sunny')(hidden_layer_2)

# Build the model
model = Model(inputs=input_layer, outputs=[regression_output, classification_output_rain, classification_output_sunny])

# Compile the model with appropriate loss functions for each output
model.compile(optimizer='adam',
              loss={'regression_output': 'mse',
                    'classification_output_rain': 'binary_crossentropy',
                    'classification_output_sunny': 'binary_crossentropy'},
              metrics={'regression_output': 'mae',
                       'classification_output_rain': 'accuracy',
                       'classification_output_sunny': 'accuracy'})

# Train the model
model.fit(X, {'regression_output': y_regression,
              'classification_output_rain': y_classification_rain,
              'classification_output_sunny': y_classification_sunny},
          epochs=10, batch_size=32, validation_split=0.2)


# construct the callback to save only the *best* model to disk
# based on the validation loss
