{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import keras\n",
    "from keras.datasets import mnist\n",
    "from keras.layers import Dense # Dense layers are \"fully connected\" layers\n",
    "from keras.models import Sequential # Documentation: https://keras.io/models/sequential/\n",
    "(x_train, y_train), (x_test, y_test) = mnist.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Training data shape:  (60000, 28, 28)\n",
      "Test data shape (10000, 28, 28)\n"
     ]
    }
   ],
   "source": [
    "print(\"Training data shape: \", x_train.shape) # (60000, 28, 28) -- 60000 images, each 28x28 pixels\n",
    "print(\"Test data shape\", x_test.shape) # (10000, 28, 28) -- 10000 images, each 28x28\n",
    "image_size = 784 # 28*28"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "image_vector_size = 28*28\n",
    "x_train = x_train.reshape(x_train.shape[0], image_vector_size)\n",
    "x_test = x_test.reshape(x_test.shape[0], image_vector_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense_21 (Dense)             (None, 32)                25120     \n",
      "_________________________________________________________________\n",
      "dense_22 (Dense)             (None, 10)                330       \n",
      "=================================================================\n",
      "Total params: 25,450\n",
      "Trainable params: 25,450\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "num_classes = 10\n",
    "y_train = keras.utils.to_categorical(y_train, num_classes)\n",
    "y_test = keras.utils.to_categorical(y_test, num_classes)\n",
    "model = Sequential()\n",
    "model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))\n",
    "model.add(Dense(units=num_classes, activation='softmax'))\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer=\"sgd\", loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "history = model.fit(x_train, y_train, batch_size=500, epochs=100, verbose=False, validation_split=.1)\n",
    "loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test loss: 0.267\n",
      "Test accuracy: 0.923\n"
     ]
    }
   ],
   "source": [
    "plt.plot(history.history['acc'])\n",
    "plt.plot(history.history['val_acc'])\n",
    "plt.title('model accuracy')\n",
    "plt.ylabel('accuracy')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['training', 'validation'], loc='best')\n",
    "plt.show()\n",
    "\n",
    "print(f'Test loss: {loss:.3}')\n",
    "print(f'Test accuracy: {accuracy:.3}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
