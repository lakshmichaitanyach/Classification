{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy\n",
    "import theano\n",
    "import theano.tensor as T\n",
    "\n",
    "class LogisticRegression(object):\n",
    "\tdef __init__(self, input, n_in, n_out):\n",
    "\t\tself.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)\n",
    "\t\tself.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)\n",
    "\t\tself.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)\n",
    "\t\tself.y_pred = T.argmax(self.p_y_given_x, axis=1)\n",
    "\t\tself.params = [self.W, self.b]\n",
    "\n",
    "\tdef negative_log_likelihood(self, y):\n",
    "\t\treturn -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])\n",
    "\n",
    "\tdef errors(self, y):\n",
    "\t\tif y.ndim != self.y_pred.ndim:\n",
    "\t\t\traise TypeError('y should have the same shape as self.y_pred', ('y', target.type, 'y_pred', self.y_pred.type))\n",
    "\t\tif y.dtype.startswith('int'):\n",
    "\t\t\treturn T.mean(T.neq(self.y_pred, y))\n",
    "\t\telse:\n",
    "\t\t\traise NotImplementedError()"
   ]
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
