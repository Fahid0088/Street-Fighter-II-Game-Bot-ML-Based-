import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv")

def correct_data (df) :
  df.columns = df.columns.str.strip()
  df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
  df = df.replace({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0})

  bool_attributes = ['round_started', 'round_over', 'player2_jumping', 'player2_crouching', 'player2_Up', 'player2_Down', 'player2_Left', 'player2_Right', 'player2_Select', 'player2_Start', 'player2_Y', 'player2_B', 'player2_X', 'player2_A', 'player2_L', 'player2_R', 'player2_in_move', 'player1_jumping', 'player1_crouching', 'player1_in_move', 'player1_Up', 'player1_Down', 'player1_Left', 'player1_Right', 'player1_Select', 'player1_Start', 'player1_Y', 'player1_B', 'player1_X', 'player1_A', 'player1_L', 'player1_R']
  for col in bool_attributes:
      df[col] = df[col].astype(int)

  string_attributes = ['result']
  for col in string_attributes:
      df[col] = LabelEncoder().fit_transform(df[col])
  print (df)
  print(df.dtypes)
  df = df.dropna()
  return df
data = correct_data(data)




input_columns = ['frame', 'timer', 'result', 'round_started', 'round_over',
              'height_delta', 'width_delta', 'player2_character',
              'player2_health', 'player2_x', 'player2_y',
              'player2_jumping', 'player2_crouching',
              'player2_in_move', 'player2_move']

output_columns = ['player1_Up', 'player1_Down', 'player1_Left', 'player1_Right',
               'player1_Select', 'player1_Start', 'player1_Y', 'player1_B',
               'player1_X', 'player1_A', 'player1_L', 'player1_R']

print (data)





X = data[input_columns]
y = data[output_columns].astype(int)

normalizer = StandardScaler()
normalized_x = normalizer.fit_transform(X)

joblib.dump(normalizer, "data.pkl")

X_train, X_test, y_train, y_test = train_test_split(normalized_x, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(y_train.shape[1], activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_history = model.fit(X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=1)

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Sample predictions:")
print(y_pred[:5])

model.save("trained_model.h5")

loss = model_history.history['loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()