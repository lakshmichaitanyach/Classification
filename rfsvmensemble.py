{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "from pylab import *\n",
    "from shutil import copyfileobj\n",
    "from six.moves import urllib\n",
    "from sklearn.datasets.base import get_data_home\n",
    "import os\n",
    "import numpy as np\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.datasets import fetch_mldata\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def fetch_mnist(data_home=None):\n",
    "    mnist_alternative_url = \"https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat\"\n",
    "    data_home = get_data_home(data_home=data_home)\n",
    "    data_home = os.path.join(data_home, 'mldata')\n",
    "    if not os.path.exists(data_home):\n",
    "        os.makedirs(data_home)\n",
    "    mnist_save_path = os.path.join(data_home, \"mnist-original.mat\")\n",
    "    if not os.path.exists(mnist_save_path):\n",
    "        mnist_url = urllib.request.urlopen(mnist_alternative_url)\n",
    "        with open(mnist_save_path, \"wb\") as matlab_file:\n",
    "            copyfileobj(mnist_url, matlab_file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fetch_mnist()\n",
    "mnist = fetch_mldata('MNIST original')\n",
    "##X, y = fetch_openml('mnist_784', version=1, return_X_y=True)\n",
    "n_train = 60000\n",
    "n_test = 10000"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "indices = arange(len(mnist.data))\n",
    "train_idx = arange(0,n_train)\n",
    "test_idx = arange(n_train+1,n_train+n_test)\n",
    "X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]\n",
    "X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "classifier2 = RandomForestClassifier();\n",
    "classifier2.fit(X_train, y_train)\n",
    "y_pred_rf = classifier2.predict(X_test)\n",
    "acc_rf = accuracy_score(y_test, y_pred_rf)\n",
    "print(\"random forest accuracy: \",acc_rf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#clf_svm = SVC()\n",
    "#clf_svm.fit(X_train, y_train)\n",
    "\n",
    "\n",
    "classifier1 = SVC(kernel='linear', C=2 , gamma=0.1);\n",
    "classifier1.fit(X_train, y_train)\n",
    "y_pred_svm = classifier.predict(X_test)\n",
    "acc_svm = accuracy_score(y_test, y_pred_svm)\n",
    "print(\"Linear SVM accuracy: \",acc_svm)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#clf_rf = RandomForestClassifier()\n",
    "#clf_rf.fit(X_train, y_train)\n",
    "\n",
    "#classifier2 = RandomForestClassifier(n_estimator=10);\n",
    "#classifier2.fit(X_train, y_train)\n",
    "#y_pred_rf = clf_rf.predict(X_test)\n",
    "#acc_rf = accuracy_score(y_test, y_pred_rf)\n",
    "#print(\"random forest accuracy: \",acc_rf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "final_pred = np.array([])\n",
    "for i in range(0,len(X_test)):\n",
    "    final_pred = np.append(final_pred, mode([y_pred_rf[i], y_pred_svm[i]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = VotingClassifier(estimators=[('svm', classifier1), ('rf', classifier2)], voting='hard')\n",
    "model.fit(x_train,y_train)\n",
    "model.score(x_test,y_test)"
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
