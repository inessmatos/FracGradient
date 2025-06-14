import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "impl")))

from frac_optimizer import FracOptimizer
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Carregar CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# One-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Construir CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Instanciar otimizador fracionário
optimizer = FracOptimizer(learning_rate=0.001, alpha=0.5)


model.compile(optimizer=FracOptimizer(), loss='categorical_crossentropy', metrics=['accuracy'])

# Treino (substitui esta parte)
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)

model.save("results/frac_cifar10_model.h5")


import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy during training with FracOptimizer")
plt.legend()
plt.grid()
plt.show()

