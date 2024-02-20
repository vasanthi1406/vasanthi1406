{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 184,
   "id": "7e3f1760",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "id": "26ec44b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix\n",
    "y_pred=np.array([0.3,0.6,0.8,0.2,0.4,0.9,0.1,\n",
    "                 0.7,0.5,0.6])\n",
    "y_true=np.array([0,1,1,0,0,1,0,1,0,1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "id": "35589b57",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 186,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy=accuracy_score(y_true,np.round(y_pred))\n",
    "accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "id": "7d1a70d7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 187,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recall=recall_score(y_true,np.round(y_pred))\n",
    "recall"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "id": "a6017585",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0"
      ]
     },
     "execution_count": 188,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "precision=precision_score(y_true,np.round(y_pred))\n",
    "precision"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "id": "7d5ba5b8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5, 0],\n",
       "       [0, 5]], dtype=int64)"
      ]
     },
     "execution_count": 189,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "matrix=confusion_matrix(y_true,np.round(y_pred))\n",
    "matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 190,
   "id": "415b7d90",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import load_iris\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "id": "3e47fd67",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'data': array([[5.1, 3.5, 1.4, 0.2],\n",
       "        [4.9, 3. , 1.4, 0.2],\n",
       "        [4.7, 3.2, 1.3, 0.2],\n",
       "        [4.6, 3.1, 1.5, 0.2],\n",
       "        [5. , 3.6, 1.4, 0.2],\n",
       "        [5.4, 3.9, 1.7, 0.4],\n",
       "        [4.6, 3.4, 1.4, 0.3],\n",
       "        [5. , 3.4, 1.5, 0.2],\n",
       "        [4.4, 2.9, 1.4, 0.2],\n",
       "        [4.9, 3.1, 1.5, 0.1],\n",
       "        [5.4, 3.7, 1.5, 0.2],\n",
       "        [4.8, 3.4, 1.6, 0.2],\n",
       "        [4.8, 3. , 1.4, 0.1],\n",
       "        [4.3, 3. , 1.1, 0.1],\n",
       "        [5.8, 4. , 1.2, 0.2],\n",
       "        [5.7, 4.4, 1.5, 0.4],\n",
       "        [5.4, 3.9, 1.3, 0.4],\n",
       "        [5.1, 3.5, 1.4, 0.3],\n",
       "        [5.7, 3.8, 1.7, 0.3],\n",
       "        [5.1, 3.8, 1.5, 0.3],\n",
       "        [5.4, 3.4, 1.7, 0.2],\n",
       "        [5.1, 3.7, 1.5, 0.4],\n",
       "        [4.6, 3.6, 1. , 0.2],\n",
       "        [5.1, 3.3, 1.7, 0.5],\n",
       "        [4.8, 3.4, 1.9, 0.2],\n",
       "        [5. , 3. , 1.6, 0.2],\n",
       "        [5. , 3.4, 1.6, 0.4],\n",
       "        [5.2, 3.5, 1.5, 0.2],\n",
       "        [5.2, 3.4, 1.4, 0.2],\n",
       "        [4.7, 3.2, 1.6, 0.2],\n",
       "        [4.8, 3.1, 1.6, 0.2],\n",
       "        [5.4, 3.4, 1.5, 0.4],\n",
       "        [5.2, 4.1, 1.5, 0.1],\n",
       "        [5.5, 4.2, 1.4, 0.2],\n",
       "        [4.9, 3.1, 1.5, 0.2],\n",
       "        [5. , 3.2, 1.2, 0.2],\n",
       "        [5.5, 3.5, 1.3, 0.2],\n",
       "        [4.9, 3.6, 1.4, 0.1],\n",
       "        [4.4, 3. , 1.3, 0.2],\n",
       "        [5.1, 3.4, 1.5, 0.2],\n",
       "        [5. , 3.5, 1.3, 0.3],\n",
       "        [4.5, 2.3, 1.3, 0.3],\n",
       "        [4.4, 3.2, 1.3, 0.2],\n",
       "        [5. , 3.5, 1.6, 0.6],\n",
       "        [5.1, 3.8, 1.9, 0.4],\n",
       "        [4.8, 3. , 1.4, 0.3],\n",
       "        [5.1, 3.8, 1.6, 0.2],\n",
       "        [4.6, 3.2, 1.4, 0.2],\n",
       "        [5.3, 3.7, 1.5, 0.2],\n",
       "        [5. , 3.3, 1.4, 0.2],\n",
       "        [7. , 3.2, 4.7, 1.4],\n",
       "        [6.4, 3.2, 4.5, 1.5],\n",
       "        [6.9, 3.1, 4.9, 1.5],\n",
       "        [5.5, 2.3, 4. , 1.3],\n",
       "        [6.5, 2.8, 4.6, 1.5],\n",
       "        [5.7, 2.8, 4.5, 1.3],\n",
       "        [6.3, 3.3, 4.7, 1.6],\n",
       "        [4.9, 2.4, 3.3, 1. ],\n",
       "        [6.6, 2.9, 4.6, 1.3],\n",
       "        [5.2, 2.7, 3.9, 1.4],\n",
       "        [5. , 2. , 3.5, 1. ],\n",
       "        [5.9, 3. , 4.2, 1.5],\n",
       "        [6. , 2.2, 4. , 1. ],\n",
       "        [6.1, 2.9, 4.7, 1.4],\n",
       "        [5.6, 2.9, 3.6, 1.3],\n",
       "        [6.7, 3.1, 4.4, 1.4],\n",
       "        [5.6, 3. , 4.5, 1.5],\n",
       "        [5.8, 2.7, 4.1, 1. ],\n",
       "        [6.2, 2.2, 4.5, 1.5],\n",
       "        [5.6, 2.5, 3.9, 1.1],\n",
       "        [5.9, 3.2, 4.8, 1.8],\n",
       "        [6.1, 2.8, 4. , 1.3],\n",
       "        [6.3, 2.5, 4.9, 1.5],\n",
       "        [6.1, 2.8, 4.7, 1.2],\n",
       "        [6.4, 2.9, 4.3, 1.3],\n",
       "        [6.6, 3. , 4.4, 1.4],\n",
       "        [6.8, 2.8, 4.8, 1.4],\n",
       "        [6.7, 3. , 5. , 1.7],\n",
       "        [6. , 2.9, 4.5, 1.5],\n",
       "        [5.7, 2.6, 3.5, 1. ],\n",
       "        [5.5, 2.4, 3.8, 1.1],\n",
       "        [5.5, 2.4, 3.7, 1. ],\n",
       "        [5.8, 2.7, 3.9, 1.2],\n",
       "        [6. , 2.7, 5.1, 1.6],\n",
       "        [5.4, 3. , 4.5, 1.5],\n",
       "        [6. , 3.4, 4.5, 1.6],\n",
       "        [6.7, 3.1, 4.7, 1.5],\n",
       "        [6.3, 2.3, 4.4, 1.3],\n",
       "        [5.6, 3. , 4.1, 1.3],\n",
       "        [5.5, 2.5, 4. , 1.3],\n",
       "        [5.5, 2.6, 4.4, 1.2],\n",
       "        [6.1, 3. , 4.6, 1.4],\n",
       "        [5.8, 2.6, 4. , 1.2],\n",
       "        [5. , 2.3, 3.3, 1. ],\n",
       "        [5.6, 2.7, 4.2, 1.3],\n",
       "        [5.7, 3. , 4.2, 1.2],\n",
       "        [5.7, 2.9, 4.2, 1.3],\n",
       "        [6.2, 2.9, 4.3, 1.3],\n",
       "        [5.1, 2.5, 3. , 1.1],\n",
       "        [5.7, 2.8, 4.1, 1.3],\n",
       "        [6.3, 3.3, 6. , 2.5],\n",
       "        [5.8, 2.7, 5.1, 1.9],\n",
       "        [7.1, 3. , 5.9, 2.1],\n",
       "        [6.3, 2.9, 5.6, 1.8],\n",
       "        [6.5, 3. , 5.8, 2.2],\n",
       "        [7.6, 3. , 6.6, 2.1],\n",
       "        [4.9, 2.5, 4.5, 1.7],\n",
       "        [7.3, 2.9, 6.3, 1.8],\n",
       "        [6.7, 2.5, 5.8, 1.8],\n",
       "        [7.2, 3.6, 6.1, 2.5],\n",
       "        [6.5, 3.2, 5.1, 2. ],\n",
       "        [6.4, 2.7, 5.3, 1.9],\n",
       "        [6.8, 3. , 5.5, 2.1],\n",
       "        [5.7, 2.5, 5. , 2. ],\n",
       "        [5.8, 2.8, 5.1, 2.4],\n",
       "        [6.4, 3.2, 5.3, 2.3],\n",
       "        [6.5, 3. , 5.5, 1.8],\n",
       "        [7.7, 3.8, 6.7, 2.2],\n",
       "        [7.7, 2.6, 6.9, 2.3],\n",
       "        [6. , 2.2, 5. , 1.5],\n",
       "        [6.9, 3.2, 5.7, 2.3],\n",
       "        [5.6, 2.8, 4.9, 2. ],\n",
       "        [7.7, 2.8, 6.7, 2. ],\n",
       "        [6.3, 2.7, 4.9, 1.8],\n",
       "        [6.7, 3.3, 5.7, 2.1],\n",
       "        [7.2, 3.2, 6. , 1.8],\n",
       "        [6.2, 2.8, 4.8, 1.8],\n",
       "        [6.1, 3. , 4.9, 1.8],\n",
       "        [6.4, 2.8, 5.6, 2.1],\n",
       "        [7.2, 3. , 5.8, 1.6],\n",
       "        [7.4, 2.8, 6.1, 1.9],\n",
       "        [7.9, 3.8, 6.4, 2. ],\n",
       "        [6.4, 2.8, 5.6, 2.2],\n",
       "        [6.3, 2.8, 5.1, 1.5],\n",
       "        [6.1, 2.6, 5.6, 1.4],\n",
       "        [7.7, 3. , 6.1, 2.3],\n",
       "        [6.3, 3.4, 5.6, 2.4],\n",
       "        [6.4, 3.1, 5.5, 1.8],\n",
       "        [6. , 3. , 4.8, 1.8],\n",
       "        [6.9, 3.1, 5.4, 2.1],\n",
       "        [6.7, 3.1, 5.6, 2.4],\n",
       "        [6.9, 3.1, 5.1, 2.3],\n",
       "        [5.8, 2.7, 5.1, 1.9],\n",
       "        [6.8, 3.2, 5.9, 2.3],\n",
       "        [6.7, 3.3, 5.7, 2.5],\n",
       "        [6.7, 3. , 5.2, 2.3],\n",
       "        [6.3, 2.5, 5. , 1.9],\n",
       "        [6.5, 3. , 5.2, 2. ],\n",
       "        [6.2, 3.4, 5.4, 2.3],\n",
       "        [5.9, 3. , 5.1, 1.8]]),\n",
       " 'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),\n",
       " 'frame': None,\n",
       " 'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10'),\n",
       " 'DESCR': '.. _iris_dataset:\\n\\nIris plants dataset\\n--------------------\\n\\n**Data Set Characteristics:**\\n\\n    :Number of Instances: 150 (50 in each of three classes)\\n    :Number of Attributes: 4 numeric, predictive attributes and the class\\n    :Attribute Information:\\n        - sepal length in cm\\n        - sepal width in cm\\n        - petal length in cm\\n        - petal width in cm\\n        - class:\\n                - Iris-Setosa\\n                - Iris-Versicolour\\n                - Iris-Virginica\\n                \\n    :Summary Statistics:\\n\\n    ============== ==== ==== ======= ===== ====================\\n                    Min  Max   Mean    SD   Class Correlation\\n    ============== ==== ==== ======= ===== ====================\\n    sepal length:   4.3  7.9   5.84   0.83    0.7826\\n    sepal width:    2.0  4.4   3.05   0.43   -0.4194\\n    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)\\n    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)\\n    ============== ==== ==== ======= ===== ====================\\n\\n    :Missing Attribute Values: None\\n    :Class Distribution: 33.3% for each of 3 classes.\\n    :Creator: R.A. Fisher\\n    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)\\n    :Date: July, 1988\\n\\nThe famous Iris database, first used by Sir R.A. Fisher. The dataset is taken\\nfrom Fisher\\'s paper. Note that it\\'s the same as in R, but not as in the UCI\\nMachine Learning Repository, which has two wrong data points.\\n\\nThis is perhaps the best known database to be found in the\\npattern recognition literature.  Fisher\\'s paper is a classic in the field and\\nis referenced frequently to this day.  (See Duda & Hart, for example.)  The\\ndata set contains 3 classes of 50 instances each, where each class refers to a\\ntype of iris plant.  One class is linearly separable from the other 2; the\\nlatter are NOT linearly separable from each other.\\n\\n.. topic:: References\\n\\n   - Fisher, R.A. \"The use of multiple measurements in taxonomic problems\"\\n     Annual Eugenics, 7, Part II, 179-188 (1936); also in \"Contributions to\\n     Mathematical Statistics\" (John Wiley, NY, 1950).\\n   - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.\\n     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.\\n   - Dasarathy, B.V. (1980) \"Nosing Around the Neighborhood: A New System\\n     Structure and Classification Rule for Recognition in Partially Exposed\\n     Environments\".  IEEE Transactions on Pattern Analysis and Machine\\n     Intelligence, Vol. PAMI-2, No. 1, 67-71.\\n   - Gates, G.W. (1972) \"The Reduced Nearest Neighbor Rule\".  IEEE Transactions\\n     on Information Theory, May 1972, 431-433.\\n   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al\"s AUTOCLASS II\\n     conceptual clustering system finds 3 classes in the data.\\n   - Many, many more ...',\n",
       " 'feature_names': ['sepal length (cm)',\n",
       "  'sepal width (cm)',\n",
       "  'petal length (cm)',\n",
       "  'petal width (cm)'],\n",
       " 'filename': 'iris.csv',\n",
       " 'data_module': 'sklearn.datasets.data'}"
      ]
     },
     "execution_count": 191,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris=load_iris()\n",
    "iris"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "id": "436a83c9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>6.7</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>6.3</td>\n",
       "      <td>2.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>6.5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>6.2</td>\n",
       "      <td>3.4</td>\n",
       "      <td>5.4</td>\n",
       "      <td>2.3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>5.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.1</td>\n",
       "      <td>1.8</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)\n",
       "0                  5.1               3.5                1.4               0.2\n",
       "1                  4.9               3.0                1.4               0.2\n",
       "2                  4.7               3.2                1.3               0.2\n",
       "3                  4.6               3.1                1.5               0.2\n",
       "4                  5.0               3.6                1.4               0.2\n",
       "..                 ...               ...                ...               ...\n",
       "145                6.7               3.0                5.2               2.3\n",
       "146                6.3               2.5                5.0               1.9\n",
       "147                6.5               3.0                5.2               2.0\n",
       "148                6.2               3.4                5.4               2.3\n",
       "149                5.9               3.0                5.1               1.8\n",
       "\n",
       "[150 rows x 4 columns]"
      ]
     },
     "execution_count": 192,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "iris_df=pd.DataFrame(data=iris.data,columns=iris.feature_names)\n",
    "iris_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "id": "cf19ced0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>sepal length (cm)</th>\n",
       "      <th>sepal width (cm)</th>\n",
       "      <th>petal length (cm)</th>\n",
       "      <th>petal width (cm)</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>5.1</td>\n",
       "      <td>3.5</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.7</td>\n",
       "      <td>3.2</td>\n",
       "      <td>1.3</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4.6</td>\n",
       "      <td>3.1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5.0</td>\n",
       "      <td>3.6</td>\n",
       "      <td>1.4</td>\n",
       "      <td>0.2</td>\n",
       "      <td>setosa</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>145</th>\n",
       "      <td>6.7</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.3</td>\n",
       "      <td>virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>146</th>\n",
       "      <td>6.3</td>\n",
       "      <td>2.5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>1.9</td>\n",
       "      <td>virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>147</th>\n",
       "      <td>6.5</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.2</td>\n",
       "      <td>2.0</td>\n",
       "      <td>virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>148</th>\n",
       "      <td>6.2</td>\n",
       "      <td>3.4</td>\n",
       "      <td>5.4</td>\n",
       "      <td>2.3</td>\n",
       "      <td>virginica</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149</th>\n",
       "      <td>5.9</td>\n",
       "      <td>3.0</td>\n",
       "      <td>5.1</td>\n",
       "      <td>1.8</td>\n",
       "      <td>virginica</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>150 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \\\n",
       "0                  5.1               3.5                1.4               0.2   \n",
       "1                  4.9               3.0                1.4               0.2   \n",
       "2                  4.7               3.2                1.3               0.2   \n",
       "3                  4.6               3.1                1.5               0.2   \n",
       "4                  5.0               3.6                1.4               0.2   \n",
       "..                 ...               ...                ...               ...   \n",
       "145                6.7               3.0                5.2               2.3   \n",
       "146                6.3               2.5                5.0               1.9   \n",
       "147                6.5               3.0                5.2               2.0   \n",
       "148                6.2               3.4                5.4               2.3   \n",
       "149                5.9               3.0                5.1               1.8   \n",
       "\n",
       "        target  \n",
       "0       setosa  \n",
       "1       setosa  \n",
       "2       setosa  \n",
       "3       setosa  \n",
       "4       setosa  \n",
       "..         ...  \n",
       "145  virginica  \n",
       "146  virginica  \n",
       "147  virginica  \n",
       "148  virginica  \n",
       "149  virginica  \n",
       "\n",
       "[150 rows x 5 columns]"
      ]
     },
     "execution_count": 193,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "iris_df=pd.DataFrame(data=iris.data,columns=iris.feature_names)\n",
    "\n",
    "iris_df['target']=iris.target\n",
    "iris_df['target']=iris.target_names[iris.target]\n",
    "iris_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "id": "a75b850d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "sklearn.utils._bunch.Bunch"
      ]
     },
     "execution_count": 194,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(iris)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "id": "b96d4970",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[5.1, 3.5, 1.4, 0.2],\n",
       "       [4.9, 3. , 1.4, 0.2],\n",
       "       [4.7, 3.2, 1.3, 0.2],\n",
       "       [4.6, 3.1, 1.5, 0.2],\n",
       "       [5. , 3.6, 1.4, 0.2],\n",
       "       [5.4, 3.9, 1.7, 0.4],\n",
       "       [4.6, 3.4, 1.4, 0.3],\n",
       "       [5. , 3.4, 1.5, 0.2],\n",
       "       [4.4, 2.9, 1.4, 0.2],\n",
       "       [4.9, 3.1, 1.5, 0.1],\n",
       "       [5.4, 3.7, 1.5, 0.2],\n",
       "       [4.8, 3.4, 1.6, 0.2],\n",
       "       [4.8, 3. , 1.4, 0.1],\n",
       "       [4.3, 3. , 1.1, 0.1],\n",
       "       [5.8, 4. , 1.2, 0.2],\n",
       "       [5.7, 4.4, 1.5, 0.4],\n",
       "       [5.4, 3.9, 1.3, 0.4],\n",
       "       [5.1, 3.5, 1.4, 0.3],\n",
       "       [5.7, 3.8, 1.7, 0.3],\n",
       "       [5.1, 3.8, 1.5, 0.3],\n",
       "       [5.4, 3.4, 1.7, 0.2],\n",
       "       [5.1, 3.7, 1.5, 0.4],\n",
       "       [4.6, 3.6, 1. , 0.2],\n",
       "       [5.1, 3.3, 1.7, 0.5],\n",
       "       [4.8, 3.4, 1.9, 0.2],\n",
       "       [5. , 3. , 1.6, 0.2],\n",
       "       [5. , 3.4, 1.6, 0.4],\n",
       "       [5.2, 3.5, 1.5, 0.2],\n",
       "       [5.2, 3.4, 1.4, 0.2],\n",
       "       [4.7, 3.2, 1.6, 0.2],\n",
       "       [4.8, 3.1, 1.6, 0.2],\n",
       "       [5.4, 3.4, 1.5, 0.4],\n",
       "       [5.2, 4.1, 1.5, 0.1],\n",
       "       [5.5, 4.2, 1.4, 0.2],\n",
       "       [4.9, 3.1, 1.5, 0.2],\n",
       "       [5. , 3.2, 1.2, 0.2],\n",
       "       [5.5, 3.5, 1.3, 0.2],\n",
       "       [4.9, 3.6, 1.4, 0.1],\n",
       "       [4.4, 3. , 1.3, 0.2],\n",
       "       [5.1, 3.4, 1.5, 0.2],\n",
       "       [5. , 3.5, 1.3, 0.3],\n",
       "       [4.5, 2.3, 1.3, 0.3],\n",
       "       [4.4, 3.2, 1.3, 0.2],\n",
       "       [5. , 3.5, 1.6, 0.6],\n",
       "       [5.1, 3.8, 1.9, 0.4],\n",
       "       [4.8, 3. , 1.4, 0.3],\n",
       "       [5.1, 3.8, 1.6, 0.2],\n",
       "       [4.6, 3.2, 1.4, 0.2],\n",
       "       [5.3, 3.7, 1.5, 0.2],\n",
       "       [5. , 3.3, 1.4, 0.2],\n",
       "       [7. , 3.2, 4.7, 1.4],\n",
       "       [6.4, 3.2, 4.5, 1.5],\n",
       "       [6.9, 3.1, 4.9, 1.5],\n",
       "       [5.5, 2.3, 4. , 1.3],\n",
       "       [6.5, 2.8, 4.6, 1.5],\n",
       "       [5.7, 2.8, 4.5, 1.3],\n",
       "       [6.3, 3.3, 4.7, 1.6],\n",
       "       [4.9, 2.4, 3.3, 1. ],\n",
       "       [6.6, 2.9, 4.6, 1.3],\n",
       "       [5.2, 2.7, 3.9, 1.4],\n",
       "       [5. , 2. , 3.5, 1. ],\n",
       "       [5.9, 3. , 4.2, 1.5],\n",
       "       [6. , 2.2, 4. , 1. ],\n",
       "       [6.1, 2.9, 4.7, 1.4],\n",
       "       [5.6, 2.9, 3.6, 1.3],\n",
       "       [6.7, 3.1, 4.4, 1.4],\n",
       "       [5.6, 3. , 4.5, 1.5],\n",
       "       [5.8, 2.7, 4.1, 1. ],\n",
       "       [6.2, 2.2, 4.5, 1.5],\n",
       "       [5.6, 2.5, 3.9, 1.1],\n",
       "       [5.9, 3.2, 4.8, 1.8],\n",
       "       [6.1, 2.8, 4. , 1.3],\n",
       "       [6.3, 2.5, 4.9, 1.5],\n",
       "       [6.1, 2.8, 4.7, 1.2],\n",
       "       [6.4, 2.9, 4.3, 1.3],\n",
       "       [6.6, 3. , 4.4, 1.4],\n",
       "       [6.8, 2.8, 4.8, 1.4],\n",
       "       [6.7, 3. , 5. , 1.7],\n",
       "       [6. , 2.9, 4.5, 1.5],\n",
       "       [5.7, 2.6, 3.5, 1. ],\n",
       "       [5.5, 2.4, 3.8, 1.1],\n",
       "       [5.5, 2.4, 3.7, 1. ],\n",
       "       [5.8, 2.7, 3.9, 1.2],\n",
       "       [6. , 2.7, 5.1, 1.6],\n",
       "       [5.4, 3. , 4.5, 1.5],\n",
       "       [6. , 3.4, 4.5, 1.6],\n",
       "       [6.7, 3.1, 4.7, 1.5],\n",
       "       [6.3, 2.3, 4.4, 1.3],\n",
       "       [5.6, 3. , 4.1, 1.3],\n",
       "       [5.5, 2.5, 4. , 1.3],\n",
       "       [5.5, 2.6, 4.4, 1.2],\n",
       "       [6.1, 3. , 4.6, 1.4],\n",
       "       [5.8, 2.6, 4. , 1.2],\n",
       "       [5. , 2.3, 3.3, 1. ],\n",
       "       [5.6, 2.7, 4.2, 1.3],\n",
       "       [5.7, 3. , 4.2, 1.2],\n",
       "       [5.7, 2.9, 4.2, 1.3],\n",
       "       [6.2, 2.9, 4.3, 1.3],\n",
       "       [5.1, 2.5, 3. , 1.1],\n",
       "       [5.7, 2.8, 4.1, 1.3],\n",
       "       [6.3, 3.3, 6. , 2.5],\n",
       "       [5.8, 2.7, 5.1, 1.9],\n",
       "       [7.1, 3. , 5.9, 2.1],\n",
       "       [6.3, 2.9, 5.6, 1.8],\n",
       "       [6.5, 3. , 5.8, 2.2],\n",
       "       [7.6, 3. , 6.6, 2.1],\n",
       "       [4.9, 2.5, 4.5, 1.7],\n",
       "       [7.3, 2.9, 6.3, 1.8],\n",
       "       [6.7, 2.5, 5.8, 1.8],\n",
       "       [7.2, 3.6, 6.1, 2.5],\n",
       "       [6.5, 3.2, 5.1, 2. ],\n",
       "       [6.4, 2.7, 5.3, 1.9],\n",
       "       [6.8, 3. , 5.5, 2.1],\n",
       "       [5.7, 2.5, 5. , 2. ],\n",
       "       [5.8, 2.8, 5.1, 2.4],\n",
       "       [6.4, 3.2, 5.3, 2.3],\n",
       "       [6.5, 3. , 5.5, 1.8],\n",
       "       [7.7, 3.8, 6.7, 2.2],\n",
       "       [7.7, 2.6, 6.9, 2.3],\n",
       "       [6. , 2.2, 5. , 1.5],\n",
       "       [6.9, 3.2, 5.7, 2.3],\n",
       "       [5.6, 2.8, 4.9, 2. ],\n",
       "       [7.7, 2.8, 6.7, 2. ],\n",
       "       [6.3, 2.7, 4.9, 1.8],\n",
       "       [6.7, 3.3, 5.7, 2.1],\n",
       "       [7.2, 3.2, 6. , 1.8],\n",
       "       [6.2, 2.8, 4.8, 1.8],\n",
       "       [6.1, 3. , 4.9, 1.8],\n",
       "       [6.4, 2.8, 5.6, 2.1],\n",
       "       [7.2, 3. , 5.8, 1.6],\n",
       "       [7.4, 2.8, 6.1, 1.9],\n",
       "       [7.9, 3.8, 6.4, 2. ],\n",
       "       [6.4, 2.8, 5.6, 2.2],\n",
       "       [6.3, 2.8, 5.1, 1.5],\n",
       "       [6.1, 2.6, 5.6, 1.4],\n",
       "       [7.7, 3. , 6.1, 2.3],\n",
       "       [6.3, 3.4, 5.6, 2.4],\n",
       "       [6.4, 3.1, 5.5, 1.8],\n",
       "       [6. , 3. , 4.8, 1.8],\n",
       "       [6.9, 3.1, 5.4, 2.1],\n",
       "       [6.7, 3.1, 5.6, 2.4],\n",
       "       [6.9, 3.1, 5.1, 2.3],\n",
       "       [5.8, 2.7, 5.1, 1.9],\n",
       "       [6.8, 3.2, 5.9, 2.3],\n",
       "       [6.7, 3.3, 5.7, 2.5],\n",
       "       [6.7, 3. , 5.2, 2.3],\n",
       "       [6.3, 2.5, 5. , 1.9],\n",
       "       [6.5, 3. , 5.2, 2. ],\n",
       "       [6.2, 3.4, 5.4, 2.3],\n",
       "       [5.9, 3. , 5.1, 1.8]])"
      ]
     },
     "execution_count": 195,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x=iris.data\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "id": "96ac9760",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
       "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
       "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])"
      ]
     },
     "execution_count": 196,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y=iris.target\n",
    "y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "id": "417877f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "id": "5b9ad88f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(120, 4)"
      ]
     },
     "execution_count": 198,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "id": "39b62e6d",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf=LogisticRegression()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 200,
   "id": "8ea7064f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-10 {color: black;}#sk-container-id-10 pre{padding: 0;}#sk-container-id-10 div.sk-toggleable {background-color: white;}#sk-container-id-10 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-10 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-10 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-10 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-10 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-10 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-10 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-10 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-10 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-10 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-10 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-10 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-10 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-10 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-10 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-10 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-10 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-10 div.sk-item {position: relative;z-index: 1;}#sk-container-id-10 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-10 div.sk-item::before, #sk-container-id-10 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-10 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-10 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-10 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-10 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-10 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-10 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-10 div.sk-label-container {text-align: center;}#sk-container-id-10 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-10 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-10\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-10\" type=\"checkbox\" checked><label for=\"sk-estimator-id-10\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 200,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.fit(x_train,y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "id": "f41a4cf5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 2, 1, 2, 1, 1, 2, 0, 2, 0, 0, 2, 2, 1, 1, 1, 0, 2, 1, 0,\n",
       "       1, 1, 1, 1, 1, 2, 0, 0])"
      ]
     },
     "execution_count": 201,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=clf.predict(x_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "id": "7ae77345",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Height</th>\n",
       "      <th>Weight</th>\n",
       "      <th>Index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Male</td>\n",
       "      <td>174</td>\n",
       "      <td>96</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Male</td>\n",
       "      <td>189</td>\n",
       "      <td>87</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Female</td>\n",
       "      <td>185</td>\n",
       "      <td>110</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Female</td>\n",
       "      <td>195</td>\n",
       "      <td>104</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Male</td>\n",
       "      <td>149</td>\n",
       "      <td>61</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>495</th>\n",
       "      <td>Female</td>\n",
       "      <td>150</td>\n",
       "      <td>153</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>496</th>\n",
       "      <td>Female</td>\n",
       "      <td>184</td>\n",
       "      <td>121</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>497</th>\n",
       "      <td>Female</td>\n",
       "      <td>141</td>\n",
       "      <td>136</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>498</th>\n",
       "      <td>Male</td>\n",
       "      <td>150</td>\n",
       "      <td>95</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>499</th>\n",
       "      <td>Male</td>\n",
       "      <td>173</td>\n",
       "      <td>131</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>500 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Gender  Height  Weight  Index\n",
       "0      Male     174      96      4\n",
       "1      Male     189      87      2\n",
       "2    Female     185     110      4\n",
       "3    Female     195     104      3\n",
       "4      Male     149      61      3\n",
       "..      ...     ...     ...    ...\n",
       "495  Female     150     153      5\n",
       "496  Female     184     121      4\n",
       "497  Female     141     136      5\n",
       "498    Male     150      95      5\n",
       "499    Male     173     131      5\n",
       "\n",
       "[500 rows x 4 columns]"
      ]
     },
     "execution_count": 202,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data=pd.read_csv(r\"C:\\Users\\GATEWAY\\Desktop\\uber\\bmi.csv\")\n",
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 203,
   "id": "d8b8985d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.26666666666666666"
      ]
     },
     "execution_count": 203,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=142)#train is used to train machinelearning functions of aix_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101\n",
    "accuracy=accuracy_score(y_test,y_pred)\n",
    "accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "id": "2fc798ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Height</th>\n",
       "      <th>Weight</th>\n",
       "      <th>Index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Male</td>\n",
       "      <td>174</td>\n",
       "      <td>96</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Male</td>\n",
       "      <td>189</td>\n",
       "      <td>87</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Female</td>\n",
       "      <td>185</td>\n",
       "      <td>110</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Female</td>\n",
       "      <td>195</td>\n",
       "      <td>104</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Male</td>\n",
       "      <td>149</td>\n",
       "      <td>61</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Gender  Height  Weight  Index\n",
       "0    Male     174      96      4\n",
       "1    Male     189      87      2\n",
       "2  Female     185     110      4\n",
       "3  Female     195     104      3\n",
       "4    Male     149      61      3"
      ]
     },
     "execution_count": 204,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "id": "58eb62af",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "245"
      ]
     },
     "execution_count": 205,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(data['Gender']==\"Male\").sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "id": "6017c133",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "255"
      ]
     },
     "execution_count": 206,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(data['Gender']==\"Female\").sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "id": "15c6a119",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender\n",
       "Female    255\n",
       "Male      245\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 207,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(data.Gender.value_counts())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "id": "4e85a56f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Height</th>\n",
       "      <th>Weight</th>\n",
       "      <th>Index</th>\n",
       "      <th>Gender_Female</th>\n",
       "      <th>Gender_Male</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>174</td>\n",
       "      <td>96</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>189</td>\n",
       "      <td>87</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>185</td>\n",
       "      <td>110</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>195</td>\n",
       "      <td>104</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>149</td>\n",
       "      <td>61</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>495</th>\n",
       "      <td>150</td>\n",
       "      <td>153</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>496</th>\n",
       "      <td>184</td>\n",
       "      <td>121</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>497</th>\n",
       "      <td>141</td>\n",
       "      <td>136</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>498</th>\n",
       "      <td>150</td>\n",
       "      <td>95</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>499</th>\n",
       "      <td>173</td>\n",
       "      <td>131</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>500 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Height  Weight  Index  Gender_Female  Gender_Male\n",
       "0       174      96      4              0            1\n",
       "1       189      87      2              0            1\n",
       "2       185     110      4              1            0\n",
       "3       195     104      3              1            0\n",
       "4       149      61      3              0            1\n",
       "..      ...     ...    ...            ...          ...\n",
       "495     150     153      5              1            0\n",
       "496     184     121      4              1            0\n",
       "497     141     136      5              1            0\n",
       "498     150      95      5              0            1\n",
       "499     173     131      5              0            1\n",
       "\n",
       "[500 rows x 5 columns]"
      ]
     },
     "execution_count": 208,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data=pd.get_dummies(data,columns=[\"Gender\"],dtype=int)\n",
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 209,
   "id": "20f2535c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Height</th>\n",
       "      <th>Weight</th>\n",
       "      <th>Index</th>\n",
       "      <th>Gender_Female</th>\n",
       "      <th>Gender_Male</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>174</td>\n",
       "      <td>96</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>189</td>\n",
       "      <td>87</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>185</td>\n",
       "      <td>110</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>195</td>\n",
       "      <td>104</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>149</td>\n",
       "      <td>61</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>495</th>\n",
       "      <td>150</td>\n",
       "      <td>153</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>496</th>\n",
       "      <td>184</td>\n",
       "      <td>121</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>497</th>\n",
       "      <td>141</td>\n",
       "      <td>136</td>\n",
       "      <td>5</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>498</th>\n",
       "      <td>150</td>\n",
       "      <td>95</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>499</th>\n",
       "      <td>173</td>\n",
       "      <td>131</td>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>500 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Height  Weight  Index  Gender_Female  Gender_Male\n",
       "0       174      96      4              0            1\n",
       "1       189      87      2              0            1\n",
       "2       185     110      4              1            0\n",
       "3       195     104      3              1            0\n",
       "4       149      61      3              0            1\n",
       "..      ...     ...    ...            ...          ...\n",
       "495     150     153      5              1            0\n",
       "496     184     121      4              1            0\n",
       "497     141     136      5              1            0\n",
       "498     150      95      5              0            1\n",
       "499     173     131      5              0            1\n",
       "\n",
       "[500 rows x 5 columns]"
      ]
     },
     "execution_count": 209,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 210,
   "id": "f105c771",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Index', ylabel='Height'>"
      ]
     },
     "execution_count": 210,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "sns.barplot(y=\"Height\",x=\"Index\",data=data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 211,
   "id": "834b9c63",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'day5' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[211], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m day5\n",
      "\u001b[1;31mNameError\u001b[0m: name 'day5' is not defined"
     ]
    }
   ],
   "source": [
    "day5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "47c6cb0b",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=data.drop(\"Index\",axis=1)\n",
    "x\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "415d5241",
   "metadata": {},
   "outputs": [],
   "source": [
    "x.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e3e0896",
   "metadata": {},
   "outputs": [],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "id": "3b392925",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 213,
   "id": "38df1b18",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 214,
   "id": "1b630a88",
   "metadata": {},
   "outputs": [
    {
     "ename": "ImportError",
     "evalue": "cannot import name 'LogisticRegression' from 'sklearn.model_selection' (C:\\Users\\GATEWAY\\anaconda3\\Lib\\site-packages\\sklearn\\model_selection\\__init__.py)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[214], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodel_selection\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m LogisticRegression\n",
      "\u001b[1;31mImportError\u001b[0m: cannot import name 'LogisticRegression' from 'sklearn.model_selection' (C:\\Users\\GATEWAY\\anaconda3\\Lib\\site-packages\\sklearn\\model_selection\\__init__.py)"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 215,
   "id": "30216311",
   "metadata": {},
   "outputs": [],
   "source": [
    "log_model=LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 216,
   "id": "28323ef7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-11 {color: black;}#sk-container-id-11 pre{padding: 0;}#sk-container-id-11 div.sk-toggleable {background-color: white;}#sk-container-id-11 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-11 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-11 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-11 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-11 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-11 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-11 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-11 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-11 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-11 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-11 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-11 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-11 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-11 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-11 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-11 div.sk-item {position: relative;z-index: 1;}#sk-container-id-11 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-11 div.sk-item::before, #sk-container-id-11 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-11 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-11 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-11 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-11 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-11 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-11 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-11 div.sk-label-container {text-align: center;}#sk-container-id-11 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-11 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-11\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-11\" type=\"checkbox\" checked><label for=\"sk-estimator-id-11\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LogisticRegression</label><div class=\"sk-toggleable__content\"><pre>LogisticRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LogisticRegression()"
      ]
     },
     "execution_count": 216,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "log_model.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 217,
   "id": "5f861821",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 2, 1, 2, 1, 1, 2, 0, 2, 0, 0, 2, 2, 1, 1, 1, 0, 2, 1, 0,\n",
       "       1, 1, 1, 1, 1, 2, 0, 0])"
      ]
     },
     "execution_count": 217,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred=log_model.predict(x_test)\n",
    "pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 218,
   "id": "b5f04034",
   "metadata": {},
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "missing a required argument: 'y_pred'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[218], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmetrics\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m accuracy_score\n\u001b[1;32m----> 2\u001b[0m accuracy_score(y_test,)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_param_validation.py:189\u001b[0m, in \u001b[0;36mvalidate_params.<locals>.decorator.<locals>.wrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    186\u001b[0m func_sig \u001b[38;5;241m=\u001b[39m signature(func)\n\u001b[0;32m    188\u001b[0m \u001b[38;5;66;03m# Map *args/**kwargs to the function signature\u001b[39;00m\n\u001b[1;32m--> 189\u001b[0m params \u001b[38;5;241m=\u001b[39m func_sig\u001b[38;5;241m.\u001b[39mbind(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[0;32m    190\u001b[0m params\u001b[38;5;241m.\u001b[39mapply_defaults()\n\u001b[0;32m    192\u001b[0m \u001b[38;5;66;03m# ignore self/cls and positional/keyword markers\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\inspect.py:3212\u001b[0m, in \u001b[0;36mSignature.bind\u001b[1;34m(self, *args, **kwargs)\u001b[0m\n\u001b[0;32m   3207\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mbind\u001b[39m(\u001b[38;5;28mself\u001b[39m, \u001b[38;5;241m/\u001b[39m, \u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs):\n\u001b[0;32m   3208\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"Get a BoundArguments object, that maps the passed `args`\u001b[39;00m\n\u001b[0;32m   3209\u001b[0m \u001b[38;5;124;03m    and `kwargs` to the function's signature.  Raises `TypeError`\u001b[39;00m\n\u001b[0;32m   3210\u001b[0m \u001b[38;5;124;03m    if the passed arguments can not be bound.\u001b[39;00m\n\u001b[0;32m   3211\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m-> 3212\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_bind(args, kwargs)\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\inspect.py:3127\u001b[0m, in \u001b[0;36mSignature._bind\u001b[1;34m(self, args, kwargs, partial)\u001b[0m\n\u001b[0;32m   3125\u001b[0m                 msg \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mmissing a required argument: \u001b[39m\u001b[38;5;132;01m{arg!r}\u001b[39;00m\u001b[38;5;124m'\u001b[39m\n\u001b[0;32m   3126\u001b[0m                 msg \u001b[38;5;241m=\u001b[39m msg\u001b[38;5;241m.\u001b[39mformat(arg\u001b[38;5;241m=\u001b[39mparam\u001b[38;5;241m.\u001b[39mname)\n\u001b[1;32m-> 3127\u001b[0m                 \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m(msg) \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m   3128\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m   3129\u001b[0m     \u001b[38;5;66;03m# We have a positional argument to process\u001b[39;00m\n\u001b[0;32m   3130\u001b[0m     \u001b[38;5;28;01mtry\u001b[39;00m:\n",
      "\u001b[1;31mTypeError\u001b[0m: missing a required argument: 'y_pred'"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "accuracy_score(y_test,)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b97185fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "#linear regression\n",
    "Data:\n",
    "    X(week)            Y(Sales in Thousand)\n",
    "    1                     1.2\n",
    "    2                     1.8\n",
    "    3                     2.5\n",
    "    4                     3.2\n",
    "    5                     3.8\n",
    "    #linear regression formula -->y=a0+a1*x\n",
    "    a0=(mean of*(x*y))-(mean of(y))/meanof(x*2)-(mean of(x)^2)\n",
    "    a0=mean(y)-a1*meanof(X)\n",
    "    a1=((meanof(x*y))-(meanof(y)))/meanof(x^2)-(meanof(x)^2)\n",
    "     x    y        x^2      x*y\n",
    "    1     1.2       1       1.2\n",
    "    2     1.8       4       3.6\n",
    "    3     2.5       9       7.5\n",
    "    4     3.2       16      12.8\n",
    "    5     3.8       25       19\n",
    "  ------------------------------------->\n",
    "sum= 15  12.5       55       44.1\n",
    "Avg= 3    2.5       11        88.8\n",
    "a1=((8.88))-(2.5))/(11-9)=0.66\n",
    "a0=2.52-(0.66*3)=0.54\n",
    "the sales of the 3rd week\n",
    "y=a0+a1*x\n",
    "y=0.54+0.66*3-->y=2.52\n",
    "the sales of the 7th week\n",
    "y=a0+a1*x\n",
    "y=0.54+0.66*7-->y=5.16"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b1400a75",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "df=pd.read_csv(r\"C:\\Users\\GATEWAY\\Desktop\\uber\\Linear_regr_Salary_dataset.csv\")\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 219,
   "id": "9495950d",
   "metadata": {},
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "\"None of [Index(['YearsExperience'], dtype='object')] are in the [columns]\"",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[219], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m x\u001b[38;5;241m=\u001b[39mdf[[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mYearsExperience\u001b[39m\u001b[38;5;124m'\u001b[39m]]\n\u001b[0;32m      2\u001b[0m y\u001b[38;5;241m=\u001b[39mdf[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mSalary\u001b[39m\u001b[38;5;124m'\u001b[39m]\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01msklearn\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mmodel_selection\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m train_test_split\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\frame.py:3767\u001b[0m, in \u001b[0;36mDataFrame.__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m   3765\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m is_iterator(key):\n\u001b[0;32m   3766\u001b[0m         key \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlist\u001b[39m(key)\n\u001b[1;32m-> 3767\u001b[0m     indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcolumns\u001b[38;5;241m.\u001b[39m_get_indexer_strict(key, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcolumns\u001b[39m\u001b[38;5;124m\"\u001b[39m)[\u001b[38;5;241m1\u001b[39m]\n\u001b[0;32m   3769\u001b[0m \u001b[38;5;66;03m# take() does not accept boolean indexers\u001b[39;00m\n\u001b[0;32m   3770\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mgetattr\u001b[39m(indexer, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdtype\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m) \u001b[38;5;241m==\u001b[39m \u001b[38;5;28mbool\u001b[39m:\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexes\\base.py:5877\u001b[0m, in \u001b[0;36mIndex._get_indexer_strict\u001b[1;34m(self, key, axis_name)\u001b[0m\n\u001b[0;32m   5874\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m   5875\u001b[0m     keyarr, indexer, new_indexer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_reindex_non_unique(keyarr)\n\u001b[1;32m-> 5877\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_raise_if_missing(keyarr, indexer, axis_name)\n\u001b[0;32m   5879\u001b[0m keyarr \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mtake(indexer)\n\u001b[0;32m   5880\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(key, Index):\n\u001b[0;32m   5881\u001b[0m     \u001b[38;5;66;03m# GH 42790 - Preserve name from an Index\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\pandas\\core\\indexes\\base.py:5938\u001b[0m, in \u001b[0;36mIndex._raise_if_missing\u001b[1;34m(self, key, indexer, axis_name)\u001b[0m\n\u001b[0;32m   5936\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m use_interval_msg:\n\u001b[0;32m   5937\u001b[0m         key \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlist\u001b[39m(key)\n\u001b[1;32m-> 5938\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mNone of [\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mkey\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m] are in the [\u001b[39m\u001b[38;5;132;01m{\u001b[39;00maxis_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m]\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m   5940\u001b[0m not_found \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mlist\u001b[39m(ensure_index(key)[missing_mask\u001b[38;5;241m.\u001b[39mnonzero()[\u001b[38;5;241m0\u001b[39m]]\u001b[38;5;241m.\u001b[39munique())\n\u001b[0;32m   5941\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mKeyError\u001b[39;00m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mnot_found\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m not in index\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "\u001b[1;31mKeyError\u001b[0m: \"None of [Index(['YearsExperience'], dtype='object')] are in the [columns]\""
     ]
    }
   ],
   "source": [
    "x=df[['YearsExperience']]\n",
    "y=df['Salary']\n",
    "from sklearn.model_selection import train_test_split\n",
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=101)\n",
    "from sklearn.linear_model import LinearRegression\n",
    "model=LinearRegression()\n",
    "model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 223,
   "id": "638214df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-13 {color: black;}#sk-container-id-13 pre{padding: 0;}#sk-container-id-13 div.sk-toggleable {background-color: white;}#sk-container-id-13 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-13 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-13 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-13 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-13 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-13 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-13 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-13 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-13 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-13 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-13 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-13 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-13 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-13 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-13 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-13 div.sk-item {position: relative;z-index: 1;}#sk-container-id-13 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-13 div.sk-item::before, #sk-container-id-13 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-13 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-13 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-13 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-13 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-13 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-13 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-13 div.sk-label-container {text-align: center;}#sk-container-id-13 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-13 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-13\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-13\" type=\"checkbox\" checked><label for=\"sk-estimator-id-13\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 223,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 221,
   "id": "652060f3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.18257493, -0.01851946,  0.24274054,  1.51014887,  1.18647929,\n",
       "        1.60639492,  1.38857942,  1.20936669,  1.71384936, -0.01382608,\n",
       "        1.75028062, -0.20437734,  0.06495741,  1.92069806,  1.59072017,\n",
       "        1.12769583,  1.1459185 ,  1.16466959, -0.01255212,  1.46076414,\n",
       "        1.02528253, -0.09999924,  1.17189317,  1.17132566,  1.14284581,\n",
       "        1.07593758,  0.9223231 ,  2.09121803,  0.01070126,  0.14320053])"
      ]
     },
     "execution_count": 221,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=model.predict(x_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 222,
   "id": "6ff1718e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 2, 1, 2, 1, 1, 2, 0, 2, 0, 0, 2, 2, 1, 1, 1, 0, 2, 1, 0,\n",
       "       1, 1, 1, 1, 1, 2, 0, 0])"
      ]
     },
     "execution_count": 222,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5184638e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "c598b369",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\GATEWAY\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:464: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([68315.11636971])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "inputdata=[[4.5]]\n",
    "prediction=model.predict(inputdata)\n",
    "prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "f6079d18",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.0333333333333334"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import mean_squared_error\n",
    "mse=mean_squared_error(y_test,y_pred)\n",
    "mse"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "1b2171e4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\GATEWAY\\anaconda3\\Lib\\site-packages\\seaborn\\axisgrid.py:118: UserWarning: The figure layout has changed to tight\n",
      "  self._figure.tight_layout(*args, **kwargs)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x16fd7e91590>"
      ]
     },
     "execution_count": 91,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 500x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "sns.lmplot(x=\"YearsExperience\",y=\"Salary\",data=df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "a66cad98",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\GATEWAY\\anaconda3\\Lib\\site-packages\\seaborn\\axisgrid.py:118: UserWarning: The figure layout has changed to tight\n",
      "  self._figure.tight_layout(*args, **kwargs)\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 500x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "sns.lmplot(x=\"YearsExperience\",y=\"Salary\",data=df)\n",
    "ax=plt.gca()\n",
    "plt.gca()\n",
    "plt.gca().set_facecolor('pink')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "89db485e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\GATEWAY\\anaconda3\\Lib\\site-packages\\seaborn\\axisgrid.py:118: UserWarning: The figure layout has changed to tight\n",
      "  self._figure.tight_layout(*args, **kwargs)\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAeoAAAHqCAYAAADLbQ06AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAB1W0lEQVR4nO3dd5xU9b34/9epU7bMbGNBpagsIoqCIEVBryhyjaIGUfMLXyMxVxPAYAkqEYxGA2KuiQYLSTRersY0VG5EjWhiEhFFsAEWcEFZEGT7bJl2ppzfH8sObJ+BLbPL+/l4+Ej2nDMznzkirz1dsW3bRgghhBBpSe3pAQghhBCibRJqIYQQIo1JqIUQQog0JqEWQggh0piEWgghhEhjEmohhBAijUmohRBCiDQmoRZCCCHSmIRaCCGESGN6Tw+gL/rl5kqseO+74ZumwA9OyeHXn1QT633D7xGyzlIn6yw1sr5S11vW2cLR+UktJ1vUIkFTFLIMDU1RenoovYass9TJOkuNrK/U9bV1JqEWQggh0piEWgghhEhjEmohhBAijUmohRBCiDQmoRZCCCHSmIRaCCGESGMSaiGEECKNSaiFEEKINCahFkIIIdKYhFoIIYRIYxJqIYQQIo1JqIUQQog0JqEWQggh0piEWgghhEhjEmohhBAijUmohRBCiDQmoRZCCCHSmIRaCCGESJICuDSlWz9TQi2EEEIkKduwyTKCdGerJdRCCCFEErINyDbLUZVIt36uhFoIIYToQJahkG2WY8UD3f7Zerd/ohBCCNGLZBoKHrOcUKwOAL17D1HLFrUQQgjRlgxdwWNUEo7VEbPtHhmDhFoIIYRohVtX8JpVWPEaona8x8YhoRZCCCGacekKXrOaiN2zkQYJtRBCCNGEU1Pwmj5ito9IPNbTw5FQCyGEEI0cmoLXrMW2q7HSINIgoRZCCCEAcKgKXrMeRakinCaRBgm1EEIIgakqeM0AqlJBKBbt6eE0IaEWQghxVDNU8JpBNLU87SINEmohhBBHMV0BrxnC0MoIxrr31qDJklALIYQ4KjVE2sLUyghE0zPSIKEWQghxFNIU8JgRHFpp0pF2aZVMHvAzJva/E0Ot7+IRHiT3+hZCCHFUURXwmlFcein+qJXUa9x6BZcMnkeOowSAff6TebdsblcOM0G2qIUQQhw1VMBjxnDppQSi4aRe49Iqm0QaoCo8tItG2JKEWgghxFFBATxmnAy9lEA0RDKP2HBplUwfMrdJpD8ov4Uvai/ssnE2J7u+hRBC9HkKkG3aZBilBKLBw470e+U38HHlD7tsnK2RLWohhBB9mgJkGzZZRinBlCLddHf3prIb+Kjie102zrZIqIUQQvRpWQZkmWUEowHiSWT6YKR3JaZtKrueDyqu68JRtk1CLYQQos/KMiDbLCcU83cc6XicY/e8yzcLr2sS6ffK/osPemBLupEcoxZCCNEnZeoKbqOcUKyOmN1+pI/d8h6j//E7+t+5Bc178IEc27d+g/e1/+rqobZLtqiFEEL0QTZZZgXhJCM9+U9LGHDnZrTjD0baesJJ4Y1bOHbLe1092HZJqIUQQvQpbl0ByonEaoja8fYXjscZ/Y/fkfXLUtTjDy7r/30evhcGY4QCnP7C0xCPQzxO3s7tHLvlPfp9tqVhWjeQXd9CCCH6DLeu4DGrAbvjSAPHfvUe/e/c0iLS/mcLAAhnZuPdu4uT1/4fx7/7L7zlX6PqA4h+Xk7VMYPZMHs+JeMmd9XXAWSLWgghRB/h0hQ8po9ovBroONJOrYpzJyxpsrv70EgDRA0TIxjkjFVPkVtSTMThIuDNxXK5KSj+lGlLFjB447qu+DoJEmohhBC9nlNT8Ji1xO1qIvFYEstXMX3wPLK8pYlp/mebRhpAt8JoVggtYuHP7UfUdIKiEnW4qCvojxmoZ8LK5V26G1xCLYQQoldzaAoesx6UKqykI30juc4vE9OsJ534f5/XbEkbV001AEFvLg23TjmEohDM9pK7q5jCbVuP8Fu0TUIthBCi1zJVBa8RQFUqCMeiHS5/MNJfJKZ9/vFFBJ7JI6OqAt0KgR1Ht0JkVJUT001ippOoYQKgGAYNl2M3nEkeNR1okQhuX2VXfD1AQi2EEKKXMlTwmkE0tYzQYUb6/fLv8U/1J7w158dUDR6KEQqQUV2BEQpQNXgoH1x1HRGXCz1ioTjdKI4C4qUhiDR8nm6FiRkGAW/zrfHOI2d9CyGE6HUaIh3C0MoIRJOL9CWDf9gs0tfxXvl/AQp7TxvL3lPPIP+Lz3HWVhPKzqHihGEAHP/uv8gt30/Q1Y/YVxbxigNbz7aNq9ZHedEISoeP7IqvCfTwFnVVVRVTp07l3XffTUxbu3Ytl112GWeccQZTpkzh0UcfJX7IQfrVq1czdepURo0axYwZM/jwww8T82KxGA888ABnnXUWo0ePZs6cOZSVlSXmV1ZWMnfuXMaOHcv48eNZsmQJ0UP+BW/evJkrr7yS0aNHM2XKFFatWtXFa0AIIUSqdAU8hoWplRGIRjpcvjHSec6diWkNkb6eJsedVZWKocP56oyJVAwdDqoKqsrHV32PaM7xuD7ag/rVVxCPo4eCZJXvx3JnsmH2/IZlu0iPhfr999/n6quvZvfu3YlpH3/8Mbfffjs333wz7733Hk888QQvvPACK1euBODdd9/lvvvuY9myZWzatIlLL72UOXPmEAwGAVixYgXr16/n+eefZ926dTidThYvXpx4/5tvvhm32826det47rnneOeddxLvXVNTww033MDll1/Opk2bWLJkCffffz9btmzptnUihBCifZoCXjOCUy/t3Ei3wVA1KkaexxuX306ZpwAz6CerohQz6Ke8aARrFz3YN6+jXr16NQsWLOCWW25pMn3v3r1861vf4rzzzkNVVU488USmTp3Kpk2bAFi1ahUXX3wxY8aMwTAMZs+eTU5ODq+88kpi/vXXX8+AAQPIzMxk0aJFvPnmm+zZs4eSkhI2btzIbbfdhsvlYuDAgcydO5dnn30WgNdeew2v18usWbPQdZ2JEycyffr0xHwhhBA9S1XAa0Zx6qX4o1aHy7ce6e8mHWlT1dAVL9VWLttHjuPPj/6ZFx78X17+6SO88OD/8udH/9zlkYYeOkY9adIkpk+fjq7rTWI9bdo0pk2blvg5FArxr3/9i+nTpwOwY8cOrrjiiibvNXToULZt20ZdXR379+9n2LBhiXn5+fl4PB62b98OgNfrpbCwMDH/xBNPZN++fdTW1lJcXNzktY3v/dxzz6X8/Uy14z8A6ahx3L11/D1B1lnqZJ2lRtZXAxXIdsTI0MoIRMMY7ayOhnVVxUWDf0iO42CkP6r4LpsrbsBQkou0quZQY+UQjdsN76lqVJ86iurGZY7oGyWvR0JdUFDQ4TL19fXcdNNNOJ1OZs+eDYDf78flcjVZzul0EggE8Pv9ALjd7hbzG+c1f23jz42vb+u9U3XjyNyUX5NOevv4e4Kss9TJOkuNrK8osA9wHvinPVXAtU0iDT9gVP7NjMpP5hceFcg98E/PS8uzvr/44gvmz59PXl4eTz/9NJmZmUBDWEOhUJNlQ6EQOTk5icg2Hq8+dH5GRga2bbeY1/hzRkYGLpeLurq6Vl+bqke3VmHFO344eboxVYUbR+b22vH3BFlnqZN1lpqjfX0pQJZhk2WUEoj6sTt4prRTqz6wJb0jMe2jitl8UH4t4Ovw8xyaDuRRY0Eo1nXXRgPcenpyl3SlXaj//e9/c+utt3LVVVfxox/9CF0/OMSioiKKi4ubLL9jxw7OOeccPB4PhYWF7NixI7ELu7y8HJ/Px7Bhw4jH4/h8PioqKsjPzwdg586d9O/fn6ysLIYNG8b69etbvHdRUVHK38GK2736P6jePv6eIOssdbLOUnO0rq9sA5x6KbURP/EkIj1t0I1NtqQ/rLiWjWXfJ5lj0k5Nx4rn47MyCce658lYyUirG5589NFHzJs3jx//+MfccccdTSINMHPmTNasWcOGDRuIRCKsXLmSyspKpk6dCsCMGTNYsWIFe/bsob6+nqVLlzJu3DgGDRrEkCFDGDNmDEuXLqW+vp49e/bw+OOPM3PmTACmTp1KRUUFK1euJBKJsGHDBtasWdPimLgQQojukWVAtllOKJZcpC8ZfGOTE8c2V1zLxrIfkEykXZpBLN4PXziTcCy9fiFKqy3qX//610SjUZYsWcKSJUsS08eMGcOTTz7JxIkTufvuu7nnnnsoLS1l6NChPPHEE3i9XgDmzZtHNBpl1qxZ+P1+xo8fz8MPP5x4n+XLl3Pvvfdy/vnno6oql19+OXPnzgUgJyeHp556iiVLlrB8+XJyc3NZvHgxEyZM6M5VIIQQAsg0FDxmOaFYHTG7o0j7WkQavs/75bNJJtJu3SAS64fPcqXlXgvFtjtYAyJlv9xcmZb/sjtiqgq3np7Xa8ffE2SdpU7WWWqOxvWVqSt4HBVYsdoOnyndEOl5LbakT8//MU995iPSwSpz6wZWrBCf5SDSzXu7F47OT2q5tNr1LYQQ4uiWoSt4zMrDjvSHFd/h/fLkdndn6CbhaH+qw90f6VSk1a5vIYQQRy+3ruA1q4jEqvHu2Nb0ntvNbtHZ2u7uDyu+w8ayOUldJ+3WHISihfgsg2ia76iQUAshhOhxDZGupmDrPznlL7/Du3cXajRKXNfxHTuEzTO+w97TxgINkb548A/Jcx68BKsx0sltSTsIHoh0mp031irZ9S2EEKJHuTQFj+mj38f/ZMIj95FbUkzE6cafk0/E6Sa3ZAeTVizj2C3vJSKd7zx4qW6ykVaADN1JMNq/10QaZItaCCFED3JqCh6zlnisklP+/CRGyI8/tx+N0Y2aTqK5DjKqyhn9+u9wXFbfLNLXpBTpQLQQn6X3mkiDhFoIIUQPcagKHrMelCqyd3yGd+8uwpkeWkZXwTomg/4Lt6K5Dj6a+KOK/8fGsrmtLN+UioJLd1IfLaTG0uhtJ8/Lrm8hhBDdzlQVPGYAVakgHIvirK1GjUaJGi0fdaFkxch8sAytqGmk3y2bR3KRdlEfGUBNuPdFGiTUQgghupmhgscMoqvlhGIN8Q1l5xDXdfRI08dXKlkxvPfvxjgxnJj2UcWs5CKtKLj0DOoi/amxFNL4Cqx2SaiFEEJ0G10Bj2FhamUEY5HE9IoThuE7dgiO+lo4cLvQViNdPot3y26k47O7FVxaFrVWP2otpYMbkKY3CbUQQohuoSngMSM4tDIC0UjTmarK5hnfIeJ0k1FVjuHw413aNNI7P7uAd8s7jrSmKEAmNZF+1EZ6d6RBQi2EEKIbqIDHjOHSywjEwq0us/e0sbw1ZyG+EUPIfuBrjKFNI/13+z46irSuqDh0D1BIfaTdRXsNOetbCCFEl1IAjxnHrZcSiIbaXbZiVBHmZfVNzu7eXP5tNtg/JJlIm1o2teEC+tJ2qIRaCCFEl1GAbNMmwyglEA22uxvaodZw8eAfUuD6PDFtc8UsNiSxu1tXVEzVQ004r889vKTv/MohhBAi7WQZkGWUETycSFd+mw1JnDhmqA2R9ll51Kf7jbsPg2xRCyFEXxGPU7htK25fJQFvHqXDR7Z4mEV3yjIUss0ygjE/8XYy3RDp+S0jXdrx7m5D1TAUDz4rF38fjDRIqIUQok8YvHEdE1YuJ3dXMVokQswwqBpSxIbZ8ykZN7nbx5OpK2SbFYRidcTtZCK9PTFtc+X/l1SkTVVDU7xUWzkE+mikQXZ9CyFErzd44zqmLVlAQfEnWK4M6vILsVwZFBR/yrQlCxi8cV23jidDV8g+8EzpWDuRNtXaNiI9n2QirSo5+Pp4pEFCLYQQvVs8zoSVyzEDddQVDCDqdIGqEnW6qCvojxmoZ8LK5RDvnvtyNT5TOmrXELXb/kxTreWSwT9sEuktld9KKtIOVUMlF1/Y2+cjDRJqIYTo1Qq3bSV3VzHB7BxQmgVOUQhme8ndVUzhtq1dPhaXpuA1fUTtGiLt/GLQEOn5LSL9TulNdBhpTQfyqLY8BHvTI7COgByjFkKIXsztq0SLRIh6HK3Oj5oOXLU+3L7KLh2HU1PwmjXE7GqseKzN5Q5GeltiWoeRjsfJ/+JzssMhgpkn8MWAQYTa2aXe10iohRCiFwt484gZBroVbtjt3YxuhYkZBgFvXpeNwaEpeM06bKUaK9Z+pC8efFNKkT52y3uc/sLT5PjrUUKZRL+qY2RuYY+dJNcTZNe3EEL0YqXDR1I1pAhXrQ+ab2XaNq5aH1VDihou1eoCDZGuR1EqCceibS7XGOl+rs8S05KJ9KQVy8it82FpA6ir0bAisR47Sa6nSKiFEKI3U1U2zJ6P5c4kq3w/eigI8Th6KEhW+X4sdyYbZs/vkuupHaqC1/SjKhWJx1W25nAiTTzO6S88jenOIOAaivV1EDsQ6LGT5HqShFoIIXq5knGTWbvoQcqLRmAG/WRVlGIG/ZQXjWDtoge7ZBexqSp4zQCqUt5BpOtaRHpr5dUdnjiW/8Xn5ASDhLSBxHZXY/v9B2d280lyPU2OUQshRB9QMm4yJWPP7pY7kzVGWlPLCHYY6fktIv126c10dHa3JxJBqXVh1dVih1o+bau7TpJLBxJqIYToK1SV0hGnd+lHGCp4zSC6Wn4Ykb4qqUi7dYN6x4lE91ShofbYSXLpQnZ9CyGESEpDpEMYWhnBWNsPe259d/dVvF16Cx1GWjMJR/uz45hTqCo8rsdOkksnEmohhBAd0hXwmmFMrYxANJlIf5qYlnykHYRj/fFZJlGl506SSzd9/xsKIYQ4IpoCXjOCo4sirQAZupNQrJBqy6DxrqA9cZJcOpJj1EIIIdp0MNKl+KNWm8u1Hukrk4q0W3cSjBbis3Sa3xW0O0+SS1cSaiGEEK1SFfCaUVx6Kf5oyzOvG5lqHd9oFumPq2bydumtdBxpF4FoIT5LI97WXUG74SS5dCahFkII0YIKeMwYLr2UQBKRLmwW6fX7f0QykfZHC6lpL9JCQi2EEKIpBcjSYxz31dso5btxZXmpOGFYi93Nplp/WJFWUXDpLvyRQmoslb5/b7EjI6EWQgjRxMmfbWTsut+R9ckG1HCIuK7jO3YIm2d8h72njQUaIz3/MCPtpi5SSK2lIBvSHTt6jsYLIYTo0PBPNvIfLyzDs3U9EcPEn5NPxOkmt2QHk1Ys49gt7x1+pBUFt0Q6ZbJFLYQQAoAszWbcuv9BK9uBP8tLY3SjppNoroOMqnJG/e0pjEvDhxVpl5ZBrdWP2ohEOhUSaiGEEGQaCkP2biCreBNhw0HL6CpYhZkMuG0LmvvgrUM/qbqiw0hrioJTy6TW6kddBIl0imTXtxBCHOUydQWPWYFatQvVX0fUMFsso7hjZD5QhnZq00i/tX8BHUc6ixqrH7US6cMioRZCiKNYhq7gMSuxYrUEMrKI6zp6pOmNTRR3DO/P9mCcHEpMSybSuqLi1LKpObAlLQ6PhFoIIY5SiUjHa4jacSpOGIbv2CE46mtp3PZV3DG8933VNNKV30wq0qaWjc8qoC4i29FHQkIthBBHIbeu4DWriNoNkQZAVdk84ztEnG4yqsrR9QDe+/ZgjAgmXrereDJvld5Oh5FWPdSE86mXSB8xCbUQQhxlXLqC16wmYtcQiTe93cje08by1pyFVA8/Hs/SfRgjDm5J7yqezNrIA7QXaUNtiLTPyqM+KpHuDHLWtxBCHEWcmoLX9BGzfUTisVaXKRt1Mvql4aZnd1d+k7cit9He9p2hahiKB5+Vi18i3Wkk1EIIcZRwqAoesx7brsZqI9KG6ucbg26mv/vjxLRPqr7JW6XtR9pUNTTFS7WVQ0Ai3alk17cQQhwFDBU8ZhBVqSDcYaS3JqZ9UvVN3trfcaRVJQefRLpLyBa1EEL0cZoCHsPC0MoIRKOtLtNapD9NItIOVUMhF1/YQ7D5w6RFp5BQCyFEH9b4TGmHVkYg2vrFzG1Fel1HkdZ0sHOptrIJSaS7jOz6FkKIPkoBsg0bl15GMNb6M6UPN9JOTQc7D59EusvJFrUQQvRRWQZkGqUEo6FWb91pqH4uGnRL00hXX55UpON2AT4rg3BjpONxCrdtxe2rJODNo3T4yBbPrxaHR0IthBB9UKahkG2WEYwFiLeS6cZID3BvSUz7tPpy1n19O+1F2qXpxOL98FluwvGG9x28cR0TVi4nd1cxWiRCzDCoGlLEhtnzKRk3udO/29FGft0RQog+xq0reIxKwrF64nZykf6s+rIkIm0Qjfejulmkpy1ZQEHxJ1iuDOryC7FcGRQUf8q0JQsYvHFdp3+/o42EWggh+hDXgRuaROzag7cGPURDpG9tEek3v76D9pLg1hsi7bPcWPGDu7snrFyOGaijrmAAUacLVJWo00VdQX/MQD0TVi6HeMtxiORJqIUQoo9wagoes5aYXd3qXcd0JXAg0psT05KNtBXrR7XlPBhpoHDbVnJ3FRPMzgGl2W1FFYVgtpfcXcUUbtuKOHwSaiGE6AMa7zqmKFWt3nVMVwJ8Y/AtzSJ9aZKRLsRnOYk02zB2+yrRIhGipqPV10ZNB1okgttXeVjfSTSQUAshRC9nqgoeM4CqVBCKtbyhSduRXki7kdZMwtH+VIcdLSINEPDmETMMdKv1S790K0zMMAh481L+TuIgCbUQQvRihgpeM4iulndypB2EY/3xWSZt3RW0dPhIqoYU4ar1QfOT1mwbV62PqiFFDZdqicMmoRZCiF7KUCHHDGFoZQRjLe86pisBvtHimPT0DiOdoTsIxQqptow2Iw2AqrJh9nwsdyZZ5fvRQ0GIx9FDQbLK92O5M9kwe75cT32EZO0JIUQv1LAlHTpw/+52Ip3xUWLaturpvPn1j2nrr34FyNCdBKOF+CyDZG44VjJuMmsXPUh50QjMoJ+silLMoJ/yohGsXfSgXEfdCeSGJ0II0cvoCnjNMGaKkf53B5F2JyKtJxXpRiXjJlMy9my5M1kXkVALIUQvoingNSM4tDL8bUT6okE/ahbpS5KItAt/tJAaSyN+OLfuVlVKR5x+GC8UHZFQCyFEL9H4JCynXoo/arWYrytBLhr0I47J+DAxrSHSd9JWpFUUXLoLf6SQGktFbk2SfiTUQgjRCyiAx4jj0ssIRFteDtUQ6VsPI9Ju6iOF1FhKqw/uED1PQi2EEL1AthHHoZcSiAZbBPWwIq0ouDU3tZFCaiXSaU2O9AshRBpruDFnjEyjjGAnRtqlZVBrSaR7A9miFkKINJZpAOwnGPO3eFzl4URaUxScWia1VgF1EYl0byChFkKINJWl2Qza9w5UVZLjU9g/ZFjikiddCfKfzU4c2+67uN1I64qKQ8uixsqnruUJ4yJNSaiFECINDf94I+Pe+h+yP98IVoDzUak+dgibZ3yH0tNP4T8H/YhjMz5ILL/ddzH/3td+pE0tG5+VT31EtqN7Ewm1EEKkmeEfb+S81Q+gV+wgbJhQUIBVFyS3ZAeT/mcp0aczyM8oTiy/3fcN/r3vTmy0Vt/PUDUMxYMvnIu/3XuCinQkoRZCiDSSqdmMW7cSvXIn/owsFEUBVSXmcOLPMsi760u0woMP32iI9KI2I22qGpripdrKISCR7pUk1EIIkSYydIUh+94la+cmwrpB4znfADjieO/eizYqtUirSg7VlpegRLrXklALIUQacOsKXrMKvXoXan0t0Zz8gzMdcTwLv8I8PZCY9NXOsfw73HakHZqOYufgC3sIpnLjbpF2evQ66qqqKqZOncq7776bmLZ582auvPJKRo8ezZQpU1i1alWT16xevZqpU6cyatQoZsyYwYcfHjzjMRaL8cADD3DWWWcxevRo5syZQ1lZWWJ+ZWUlc+fOZezYsYwfP54lS5YQjUaT/mwhhOgKLl3Ba1YTsWvwuzOJ6zp65MAtQh1xuHN3k0hH/ubk3U/mthlpp6aDnUe1JZHuC3os1O+//z5XX301u3fvTkyrqanhhhtu4PLLL2fTpk0sWbKE+++/ny1btgDw7rvvct9997Fs2TI2bdrEpZdeypw5cwgGgwCsWLGC9evX8/zzz7Nu3TqcTieLFy9OvP/NN9+M2+1m3bp1PPfcc7zzzjusXLkyqc8WQoiu4NQUvEYNMdtHJB6j4oRh+I4dgqO+FhwxPPd8BSP9ieUjr5jsXzmSihOGt/F+OnE7n2ori5BEuk/okVCvXr2aBQsWcMsttzSZ/tprr+H1epk1axa6rjNx4kSmT5/Os88+C8CqVau4+OKLGTNmDIZhMHv2bHJycnjllVcS86+//noGDBhAZmYmixYt4s0332TPnj2UlJSwceNGbrvtNlwuFwMHDmTu3LmJ9+7os4UQorM5NAWvWYdNFVY81jBRVdk84ztEPE7yFn3ZdEv6FRP/fxew+ZvXtvoIyYZIF+CzMglLpPuMHjlGPWnSJKZPn46u601iXVxczLBhw5osO3ToUJ577jkAduzYwRVXXNFi/rZt26irq2P//v1NXp+fn4/H42H79u0AeL1eCgsLE/NPPPFE9u3bR21tbYefnQpTVTpeKA01jru3jr8nyDpLnayzBo2RhkrCsRjGIaujctSpRJ/JRCvcl5gW+ZuT0v8dydY511I2cixGs/driHQ/6iIZ2LZ9VK/fvvZnrEdCXVBQ0Op0v9+Py+VqMs3pdBIIBDqc7/c37Bpyu90t5jfOa/7axp8bX9/eZ6fixpG5Kb8mnfT28fcEWWepk3VWA0SArGbTg8AtwOcHJ5VOwBh4K8f9cSTHtbIlDRpQCGR2zVB7qb7yZyytzvp2uVzU1dU1mRYKhcjIyEjMD4VCLebn5OQkItt4vLr5623bbjGv8eeMjIwOPzsVj26twjqsJ6/3LFNVuHFkbq8df0+QdZa6o32dOVQFj6MehXLCsWiTeZoSYurABRyT8V5i2hc1/8kJhb/k9746rO01Ld7PpRtE4/3wWWEi8ZaPvzwa9ZY/Y7eenpfUcmkV6mHDhrF+/fom03bs2EFRUREARUVFFBcXt5h/zjnn4PF4KCwsZMeOHYld2OXl5fh8PoYNG0Y8Hsfn81FRUUF+fsNlDzt37qR///5kZWV1+NmpsOJ2Wv/h6EhvH39PkHWWul61zuJxCrdtxe2rJODNo3T4yFaPEXfEoSm4jTpidgWhZpHWlRDnH3dbk0gX+6bx1td3cYJHw4rbNL/zp1s3CEb74bNcvWdddqNe9WesHWn1mMupU6dSUVHBypUriUQibNiwgTVr1iSOS8+cOZM1a9awYcMGIpEIK1eupLKykqlTpwIwY8YMVqxYwZ49e6ivr2fp0qWMGzeOQYMGMWTIEMaMGcPSpUupr69nz549PP7448ycOTOpzxZCHJ0Gb1zH1TdezYwF13Lx3T9kxoJrufrGqxm8cV1K79NwTLoeVWk90tMG3sZxmZsS04p90/jnvp+0eQmWWzewYv2otpx9IkaibWm1RZ2Tk8NTTz3FkiVLWL58Obm5uSxevJgJEyYAMHHiRO6++27uueceSktLGTp0KE888QRerxeAefPmEY1GmTVrFn6/n/Hjx/Pwww8n3n/58uXce++9nH/++aiqyuWXX87cuXOT+mwhxNFn8MZ1TFuyADNQRzA7h6jHgW6FKSj+lGlLFrB20YOUjJvc4fs0RNqffKRrLmw/0ppJOFpITcQkEj+y7yjSn2Lbtvwq1sl+ubmyV/6Ga6oKt56e12vH3xNknaWu16yzeJyrb7yaguJPqCsYAMohZxDbNlnl+ykvGsGfH/1zu7vBHaqC1+FHVcpbRFpTQvxna5Hee3ci0oYC152cw1OfVRO1wa07CEYLqbEM5K6grestf8YWjs7veCHSbNe3EEKki8JtW8ndVUwwO6dppAEUhWC2l9xdxRRu29rmexxppJt8JJChOwlG++OTSB9VJNRCCNEKt68SLRIhajpanR81HWiRCG5fZavzHaqC1wx0SqQB3Iab+mh/qi0duZfJ0UVCLYQQrQh484gZBrrV+iVPuhUmZhgEvC0vsTEPRFpTy1qN9LSBt7cS6daPSasogJv6SH9qwhppvCdXdBEJtRBCtKJ0+EiqhhThqvVB81N5bBtXrY+qIUUNl2od4tBIB1uN9B0MzNyYmHYw0i3P7VUVBaeeCfSn1lKR88aOThJqIYRojaqyYfZ8LHcmWeX70UNBiMfRQ0GyyvdjuTPZMHt+kxPJGiOttxvpg08L7CjSLi2T+kg/QEc2pI9eEmohhGhDybjJrF30IOVFIzCDfrIqSjGDfsqLRrS4NOtgpMuTivSOmqltRlo7EOlaqx+1kb5xv2px+NLqOmohhEg3JeMmUzL27HbvTNYQ6eCBSEeavL6tSL+x9+42I+1MRBpM2Zw66kmohRCiI6pK6YjTW51lqOA1gxhaGYFo50S6xupHXaTFbHGUklALIcRhaoh0KIVIX9BBpLOosQok0qIJ2akihBCHoTHSZquRDjNt4MJWIn2PRFqkTEIthBAp0hXwmuF2In0HAzM3JKZJpMWRkFALIUQKGiJtYWqlSUV6Z835HUQ6W45Ji3bJMWohhEiSpoDXjODUS/EnGel/7P1pq5HWFRVHYktarpIWbZNQCyFEEhoj7dBK8UetZvPCXJhipE0tG5+VT71EWnRAQi2E6DnxeLvXJ6cLVQGvGcWll+GPNr33d2OkB6UY6RqJtEiShFoI0SMGbnyTsU8tJ3dXMVokQswwqBpSxIbZ85vc8aunqQp4zBguvZRANNRk3mFFWvVQE86jXp5TKZKUfr+6CiH6vjfe4IKfLaCg+BMsVwZ1+YVYrgwKij9l2pIFDN64rqdHCDT8BekxY2QciPShadWUMBcet7BFpN9oI9KG2hBpnyWRFqmRUAshulc8DsuWYfrrqCsYQNTpAlUl6nRRV9AfM1DPhJXLG5brQQrgMeMHIh1sPdJZ7ySmNUY63mqkNQzFi8/Kwy+RFimSUAshulW/bVtg+3aCnhxQmj1wQlEIZnvJ3VVM4batPTNAGiKdbdpkGK1HeupxP24a6dopbUbaVDV0xUu1lSuRFodFQi2E6FYuXxVYFlHT0er8qOlAi0Rw+yq7eWQHZRs2WUYpwWaRVhWLqcf9mMFZbyem7aydwhtf3dtmpDUlB5+VQ0AiLQ6ThFoI0a2C3lwwTXQr3Op83QoTMwwC3rxuHlmDLAOyzHKCsQDxQzKtKhYXHrcwpUirSg7VllciLY6IhFoI0a3Khp8GJ52Es8YHdrOA2TauWh9VQ4oaLtXqZpm6gsesIBSrJ263H+kvas9rM9IOVUMlF1/YS1AiLY6QhFoI0b1UFRYuJJKRSVb5fvRQEOJx9FCQrPL9WO5MNsye3+3XU2foCh6zknCsjliLSP+4RaT/8dV9rUbaqelAHtWWh2BMIi2OnIRaCNH9pkzh74sfpLxoBGbQT1ZFKWbQT3nRCNYuerDrr6OOxyn8dDPHv/0GhZ9uxqXaeMxqInYtUfvg2eYHI70+Ma2jSMftfHxWNiGJtOgkcsMTIUSP2DPuHHaecXa335ls8MZ1TFh58EYr8YIC6s46la3fOJ+Sk09LLNd6pP+jg0gX4LMyCEukRSeSUAsheo6qUjri9G77uMEb1zFtyQLMQB3B7Bxix+ZhHuPGu/s9JixfT3TOQvaeNrbVSH9Zey7/+OpnrUbapenE4v3wWW7CcYm06Fyy61sIcXSIx5mwcjlmoOFGK7HcPLTBucTcQfy6jhEKcPoLT6PaoVYj/fevlrQRaYNovB/VEmnRRWSLWghxVCjctpXcXcUEs3NQMjLQBuWiOGqIV1cBCuHMbLxlX3Jx3s0ck/VR4nUNkW5rS9ogGi/AZ7mxJNKii0iohRBHBbevsuHhHwO8DZF21hCvrkjMj7p1sm/zkdW/PDHtYKSNlu+nG0Ri/fBZLom06FISaiHEUSHgzSPm8WD2zyLurG0SaYw43oV70c+KJCYlE+lqy0lEIi26mByjFkIcFapGjKTurDE47X3Eq8sOztBtPHfuxTwrmJiUfKS7Y+TiaCehFkL0eYYKXqfFp1dciBWoI6OqHN0KgRbD++PdOCb4E8tKpEW6kVALIfo0QwWvGcLUyvhy2AjemrOQqsFDMaJ+cm/fJVvSIu3JMWohRJ91aKQD0Ybjz3tPG8vXI0dyce7NZA1I7cQxibToCRJqIUSf1FqkAVQiXDDwLo7J/jAx7cvac9qNtBXrh08iLXqIhFoI0ec0RDrceqSPW8Tx2W8mpjVEekk7kS7EZzkk0qLHSKiFEH3KwUiXthLpxUlHOkM3CUUL8Vkm8qRK0ZPkZDIhRJ/RcaT/nZi2q25y21vSmkRapA/ZohZC9Am6kkqkJ/H6nqVtRNpBOFaIzzIk0iItyBa1EKLXa4i0lUKk729jd7eDUKyQaom0SCOyRS2E6NUaI+3US/EfZqQVwK07CUb74bMM5HHSIp1IqIUQvVZDpCMHIm0lpqtEW4/0Vy13dx+MdCE+S5dIi7Qju76FEL2SdiDSDm1/K5Fe1HqkbbPJe0ikRW8gW9RCiF7nYKRLCcTaj3RJ3dltRjpDd+KP9sdnachDsES6ki1qIUSv0hhpl15KIBZOTFeJcn6z3d0ldWfz2lf3t7El7aJeIi16AdmiFkL0GqoCXjOKSy/DH20Z6ROy/5WY1lakVRRcugt/pJAaS0VuOCbSnYRaCNErKIDHiOPSywhEQ4npDZG+K6VI10f6U2spEmnRK0iohRBpTwGyTZsMo5RANEjjnuqDkf5nYtn2I+2mLlJIraUge7tFbyHHqIUQaU0Bsg2bLKOUYLNITznuJ8lFWlFwS6RFLyVb1EKItJZlQJZZRjAWIH4gsY2RPjH7jcRy7UXapWVQa/WjNiKRFr2PhFoIkbayDIVss4xQzE/cbi/SZ7UaaU1RcGqZ1FoF1EmkRS8loRZCpKVMXSHbrCAcqyfWYaSXtRPpftRGEKLXklALIdJOhq6QbVZixWqJ2g3nZqtEmXLs3S0i/XqbW9JZ1FgF1EmkRS8noRZCpJUMXcFjVhK1a1pG2vOPxHKNkY7Zjiav1xUVRyLSsrNb9H4pn/W9Z8+erhiHEEKQqSt4zUoi8Roi8bYjvbtuYpuRNrVsfBJp0YekHOqLLrqIa665hr/+9a+EQqGOXyCEEEnIPLAlbcUPbkkrbUT6ta+WtRnpGiufeom06ENSDvW///1vzjvvPH73u98xadIk7rrrLj788MOuGJsQ4iiRaSh4HBUtIn1+80jXT2g70qqHmrBEWvQ9KYc6Ly+P6667jhdffJGnn36a7OxsFi5cyEUXXcSTTz5JVVVVV4xTCNFHZRkKXrO8yYljbUZ6zwMtIm2oDZH2WXnURyXSou857DuTRaNR9u3bx759+6isrMTlcrF582YuvPBCVq9e3ZljFEJ0p3icwk83c/zbb1D46WaId90dsbMM8JhlhDqI9J42I61hKF58Vh5+ibToo1I+6/ujjz7ir3/9K3/7299QFIXp06fz+9//nuHDhwPw+uuvs2jRIr75zW92+mCFEF1r8MZ1TFi5nNxdxWiRCDHDoGpIERtmz6dk3ORO/awsA1x6GaFDrpNuOCZ9T4tIr20z0h58Vq5EWvRpKYd61qxZTJo0iZ/+9KdMmTIFwzCazD/55JOZMmVKpw1QCNE9Bm9cx7QlCzADdQSzc4h6HOhWmILiT5m2ZAFrFz3YKbFWAIiRZZRRF61P3HGsMdJDPX9PLNtepHXFS7WVQ0AiLfq4lEM9Z84cvvvd75KRkdHq/OOOO45ly5Yd8cCEEN0oHmfCyuWYgTrqCgaA0pDTqNNFncNJVvl+JqxcTsnYs0E9/Gf5KECWYQP7CcXqDivSpqqhKV58EmlxlEj5v7inn34al8vVFWMRQvSQwm1byd1VTDA7JxHpBEUhmO0ld1cxhdu2HvZnND4FK9MoAwIdRHp8O5HOkUiLo0rKoZ48eTK//e1vKSsr64rxCCF6gNtXiRaJEDUdrc6Pmg60SAS3r/Kw3j/xqEqz4Zg0HBrpnyYVaYeqoZJLteWVSIujSsq7vt9//31efvllfvWrX7WY99lnn3XKoIQQ3SvgzSNmGOhWmKiz5R4z3QoTMwwC3rzDev+sA5EOxvxoLSL9emK5g5F2Nnm9Q9PBzqXayiYUk0iLo0vKof75z3/eFeMQQvSg0uEjqRpSREHxp9Q5nE13f9s2rlof5UUjKB0+MuX3zjYg+0Ck47EY/XZ9DhVhpp7wPxzn2ZBYbk/9uFYj7dR0bDsPn5UlkRZHpZRDPW7cuFany41OhOjFVJUNs+czbckCssr3E8z2EjUbzvp21fqw3JlsmD0/5RPJDo30gM2bOP2Fp8nZ/yUsruW4Y8OJ5Roi/fNWIx238/FZmYQl0uIolXKot2zZws9//nNKS0uJH7gRQiQSoaqqio8//rjTByiE6B4l4yazdtGDieuoXbU+YoZBedGIw7qOOutApEOxegZs3sSkFcswIvXo90bggoORjmxysHXXt4idIpEWojUph/ree+9l4MCBFBUVsWfPHs4++2yefvppfvSjH3XF+IQQ3ahk3GRKxp5N4batuH2VBLx5Dbu7U9ySzjQUPGZ5w81MYjFOf+FpDKsebWkMx3n+xHLWB24iP3Iwsv+f2HPyhMTnSKSFOCjls76Li4u5//77mTVrFrFYjO9+97s89NBDrFmzpivGJ4TobqpK6YjT+fKsKZSOOD31SOsKHrOCcKyOmG2T/8XneL/+Ev2+CM7z6hLLWR+68f30OMKGB+/eXeR/8TkgkRaiuZRDnZ2djdPpZODAgRQXFwMwatQo9u7d2+mDE0L0LhmNj6o85N7dzrpKXItqcFxwcEuazRnU/PQ4sFSihokajeKsrZZIC9GKlEN9wgkn8Mc//hGHw4Hb7eazzz5j586dKM1vknAEPvnkE2bNmsXYsWOZNGkSP/vZz7AsC4DNmzdz5ZVXMnr0aKZMmcKqVauavHb16tVMnTqVUaNGMWPGjCaP4IzFYjzwwAOcddZZjB49mjlz5jS5HryyspK5c+cyduxYxo8fz5IlS4hGo532vYToyzJ0Ba9ZScQ+9FGVMUaeuwr9P63EctaHbrh/EFgNf/3oEYu4rhPP7y+RFqIVKYf6pptu4uGHH2b37t1873vf46qrruKKK67otIdwxONxvv/97zNt2jQ2btzIc889x1tvvcUTTzxBTU0NN9xwA5dffjmbNm1iyZIl3H///WzZsgWAd999l/vuu49ly5axadMmLr30UubMmUMwGARgxYoVrF+/nueff55169bhdDpZvHhx4rNvvvlm3G4369at47nnnuOdd95h5cqVnfK9hOjL3LqCx6wmYtcSiR+M9HnH3Mug4w5egmV94Kbm3uMSkQYbR30ttcNHUTF4vERaiFakHOozzjiDN998k+OOO46rr76aZ599lscee4w77rijUwZUU1NDeXk58Xgc+8AtBlVVxeVy8dprr+H1epk1axa6rjNx4kSmT5/Os88+C8CqVau4+OKLGTNmDIZhMHv2bHJycnjllVcS86+//noGDBhAZmYmixYt4s0332TPnj2UlJSwceNGbrvtNlwuFwMHDmTu3LmJ9xZCtM6lK3hNHzHbRyQeAw5Gusi7NrFc5D0HkZsdaLUWxONo4RAZVeVE+w9k6zfm4otkS6SFaEXSoW589nTj86f379/Pvn37yM/P5/jjj2ffvn2dMqCcnBxmz57NAw88wMiRIzn33HMZMmQIs2fPpri4mGHDhjVZfujQoWzbtg2AHTt2tDm/rq6O/fv3N5mfn5+Px+Nh+/btFBcX4/V6KSwsTMw/8cQT2bdvH7W1tZ3y3YToa1xaY6SrsQ6J9H8cc1+TSH9VP5bXv7yfqv5FmKEAlJdjhgL4ThvHO9/5KZ8NnSSRFqINSV+eNWXKlMRxaNu2mxyTbvy5M24hGo/HcTqd3HXXXcycOZOSkhJuvPFGli9fjt/vb/FAEKfTSSAQAGh3vt/fcCKL2+1uMb9xXvPXNv4cCATIzs5O+juYaucdr+9OjePurePvCUfzOnNqCh6zBtuuJmbHMJSGSE8+5j6Gel5NLLfPP5Z/fPUgsVOd7D9lIoVfbufi7CjvhDIpPe4sfNEM7Lh9VK7DZBzNf8YOV19bZ0mH+h//+EfHC3WC119/nbVr1/Lqqw3/oRcVFTFv3jyWLFnC9OnTqaura7J8KBRKPHLT5XIRCoVazM/JyUlEt/F4dfPX27bdYl7jz2090rMtN47MTWn5dNPbx98Tjs515gNiQOMvsTFgIfDqIctM4JiMX3Pt8EN+CR4xEdCYSiGQ2S0j7QuOzj9jR6avrLOkQ33ssce2OS8ajfL555+3u0yyvv7668QZ3o10XccwDIYNG8b69eubzNuxYwdFRUVAQ9QbLxk7dP4555yDx+OhsLCwye7x8vJyfD4fw4YNIx6P4/P5qKioID8/H4CdO3fSv39/srKyUvoOj26twor3vt14pqpw48jcXjv+nnA0rjNTVfCaflSljFCs4aoIhRiTB/yMod6/JZbb5x/L63uWEbNDwMFfoLMNk5lDi/jNJxb+6OE9jetocjT+GTtSvWWd3Xp6cg+5SfnOZP/617/46U9/SmlpaeJkL2iI6dath/+s2kaTJk3iF7/4Bb/+9a+5/vrr2bdvHytWrGD69OlMnTqV//7v/2blypXMmjWL999/nzVr1vD4448DMHPmTObNm8dFF13EmDFjePbZZ6msrGTq1KkAzJgxgxUrVjBy5EhycnJYunQp48aNY9CgQQCMGTOGpUuXcu+991JdXc3jjz/OzJkzU/4OVtxO6z8cHent4+8JR8s6c6gKGbqfGGX4owcj/R/HNI30Xv9YXt39INFm9+52aQahWAGQiT9aeVSss85ytPwZ60x9ZZ2lHOoHH3yQCy+8kOzsbLZv384ll1zCY489dlhBa83QoUP5zW9+w8MPP8yTTz5JVlYWl156KfPmzcM0TZ566imWLFnC8uXLyc3NZfHixUyYMAGAiRMncvfdd3PPPfdQWlrK0KFDeeKJJ/B6vQDMmzePaDTKrFmz8Pv9jB8/nocffjjx2cuXL+fee+/l/PPPR1VVLr/8cubOndsp30uI3s6hNW5JlzfZkv6PY37GsKQirRONFxCINj1PRAjRvpRDvWfPHm677Ta++uorNmzYwIUXXsgJJ5zALbfcwjXXXNMpgzrrrLM466yzWp03cuRI/vSnP7X52ssuu4zLLrus1XmGYbBgwQIWLFjQ6vz8/HyWL1+e+oCF6OMaThyrR1UqmkT63GOWNIv0mHYi3Q+f5QZ6/xaOEN0p5euoc3NzUVWVY445hp07dwINW8H79+/v9MEJIXqeU1PwmrUorUT6JO8rieUaIv2LdiPdF3ZDCtHdUg71SSedxK9+9SsA8vLy+Pe//827776Lw+Ho9MEJIXpWY6ShknCHkW65Je3UdGISaSGOSMqhvu222/j73/9OeXk58+fPZ+7cucyePZvvfe97XTE+IUQPaRLpQ25m0jLSZxyIdLN7GGg6cYm0EEcspWPU8Xic3NxcXn75ZaDhePXcuXO54IILOOmkk7pkgEKI7peItFJFONZRpH/RbqTDEmkhjkjSW9SlpaVMnz6dn//85wCsWbOG6667jn/84x/MmjWrUy7NEkL0vIZI1x2I9KG7u5cmFWmXRFqITpV0qB966CFOOumkxBnTjzzyCNdffz0vvPACP/nJT3jkkUe6bJBCiO7hSES6+THppZzkfTmx3D7/6DYjHYv3o1oiLUSnSTrU69evZ/HixeTl5bFv3z52797NpZdeCsD555/PRx991FVjFEJ0g4ZI16McEmmIc86A+1tE+m+7f9nq7u7GSMsxaSE6T9Khrq+vJze34b6pmzdvJjs7mxNPPBEAh8NBJBLpmhEKIbqcI3Fb0IOXYEGccwcsZXjOS4nl2ou0nDgmRNdIOtQej4eqqioANm7cyBlnnJGY98UXX5CTk9P5oxNCdLmGe3cHmtxx7HAjLbu7heh8SYf6vPPO47777uOVV15hzZo1XHzxxQDU1tbyq1/9ismTJ3fZIIUQXaMx0ppadviRtgsk0kJ0oaRDfcstt1BTU8Odd97JtGnTmD59OgDnnnsuxcXF/PCHP+yyQQohOp+hgtcMoqvlBJtE+v5WIt3GJVh2Ab5whkRaiC6U9HXU2dnZPPXUUy2mP/LII5x55plyZzIhepGGSIcwtDIC0cbzSxpOHBuesyax3Nf+UQci3fRBGhJpIbpPyg/laG7SpEmdMQ4hRDdpjLTZSqRPbhbpV3b/UiItRA874lALIXoPQ4WcNrakm0Q6cLpEWog0kfK9voUQvZOpKm1EelnLSJc8JJEWIk1IqIU4CjSc3R1sI9IvJpbrMNKWRFqI7iahFqKPa4y0rh5JpPMbIh2TSAvR3eQYtRB9WON10g2XYB0a6QdaRPpvbR6TzsdnZUqkheghskUtRB/VVqQnD/g5J+f8NbHc14HT+NvuXxKJZzR5vUPTse08ibQQPUxCLUQf1F6kR+T8X2K5hkg/1GqksfOotrIk0kL0MAm1EH1MspHeHxjZeqRVDexcfBJpIdKCHKMWog85GOmyJrcFbS3Sr+x+uPVIk4fPyiYkkRYiLcgWtRB9hERaiL5JQi1EH9AZkVbIlUgLkYYk1EL0ck2PSR8S6f7/nVSkzQORrrY8Emkh0pCEWoherM0Tx/r/NyNyVyeWay/SqpKDTyItRNqSUAvRSx2841jTSE9KMtKGqqEpXmosL0GJtBBpS0ItRC906G1Bm0f6lKQirWIoHnxWDoGoRFqIdCahFqKXSTbSpYFTO4h0rkRaiF5ArqMWohdJJdIv7/5Vq7u7GyPtl0gL0StIqIXoJdqO9INJbUmbB45JV8vubiF6FQm1EL1A62d32wci/UJiudLAKbyy+2GseGaz1zdEWo5JC9H7SKiFSHOOA5HWmtzMxD6wu7t5pH8lkRaij5FQC5HGHJqC1/SjKuXNIt3alrREWoi+SEItRJpyagoesx5VqSDUItLPJ5aTSAvRt0mohUhDTk3Ba9aCUtUk0mf3/0VSkTYk0kL0GRJqIdKMS1fwmj5su5pwLHZgakOkT819LrFcW5HWFRVDyZazu4XoIyTUQqQRt67gNauJ2T6s+OFF2lQ9+Kw8ibQQfYTcmUyINJGhK+SYVUQ7inRwRKuR1hQFh5ZFTSRPbmYiRB8iW9RCpIFMXcFjVhKxa4jE4wemthHpktYj7dSyqLEKqA/HKNy2FbevkoA3j9LhI0GV38mF6K0k1EL0sExDwWNWYMVqidoHI31W4S/biHRWk9cfGunc9W/yjZXLyd1VjBaJEDMMqoYUsWH2fErGTe7GbyWE6Czya7YQPai9SI/MW5VYLrlIr2PakgUUFH+C5cqgLr8Qy5VBQfGnTFuygMEb13XjNxNCdBYJtRA9JNlIl7Ub6UxqrALqwnEmrFyOGaijrmAAUacLVJWo00VdQX/MQD0TVi6HxG51IURvIaEWoge0HemHW0T65VYiraLg1DKotfpRF4HCbVvJ3VVMMDsHFKXphykKwWwvubuKKdy2tYu/mRCis0mohehm7Uf6z4nl2ou0S3dTdyDSAG5fJVokQtR0tPqZUdOBFong9lV2xVcSQnQhCbUQ3SjLUPCa5UlE+uRWI60ALt1FfaSQ2ohC40VYAW8eMcNAt8Ktfq5uhYkZBgFvXud/KSFEl5JQC9FNsgzwmGWEjiDSbt2FP1JIrXUw0gClw0dSNaQIV60P7GbXUNs2rlofVUOKGi7Viscp/HQzx7/9BoWfbpbj1kKkObk8S4hu0BDpckKxOmKJkNpMLPxVG5HObvL6hkg7CUQLqYmotEirqrJh9nymLVlAVvl+gtleoqYD3QrjqvVhuTPZMHs+g99bzwS5fEuIXkW2qIXoQgqQndiSbhnp0/L+lFi2rUgDuHUHwWghNZZGvI2bjpWMm8zaRQ9SXjQCM+gnq6IUM+invGgEaxc9CCCXbwnRC8kWtRBdpCHSNllmGcGYn3g7kS4PDm870lpDpH2WTqyDO4OWjJtMydizW96ZDLj6xqsTl281nhkedbqoczjJKt/PhJXLKRl7ttzFTIg0I6EWogsoQLZpk2UkF+mXSpa3GukM3SQULcRnGR1GOkFVKR1xepNJhZ9uTvryreavFUL0LPnVWYhOdjDSpQSjhx9pt24QjvVLLdJtkMu3hOi9JNRCdCIV8Jg2WcZ+gtEAcQ5GekLh8maRPqndSFuxfvgsB53xICy5fEuI3ktCLfqWHrz0qCHScTKN/QSjwRaRPj3vj4lly4Mn8XIbkXZpOpFYATWWi0gnDT+ly7eEEGlFjlGLPmPwxnU9dumRqoDHESNDLyUQDR5yjXPbkQ7HPS3ex6npxOwCfJYbq63Tuw9rgMldviUnkgmRfuS/StEnDN7Ys0+OyjbbivQjSUfaoWrYdh4+K6NzI31AR5dvyXXUQqQn2aIWvV+86ZOjuvPSI00BCJOh7acm0lqk/5CY0l6kTVVDUXKoDmcRPtIzx9rR5uVbsiUtRNqSUIteL5UnR3XmpUeaAh4zCuxvuSXd79FmkR7WZqQNVUVTvFRbXkJdGOmEVi7fEkKkLwm16PUSlx552r70yFXr69RLjzQFvLrFgK/egfIacms09g8ZBqrSEOn8ZxPLNkT6kVYjrSsqhuLBZ+UQ7IzTu4UQfY6EWvR6h156FHW6Wszv7EuPdAVGfL6BUW/8huyPN4Ed43xUqo8djPVTDyfm/z2xbEeRNlUPPisPv0RaCNEGCbXo9RovPSoo/pQ6h7Pp7u8Dlx6VF43olEuPdAVGFG/g7D/fi753J+GMbMhyYdUFKJy+FfPkUGLZio4irWVTE5ZICyHaJ2eQiN7vwKVHljuTrPL96KEgxOPooSBZ5fs77dIjQ4UcM8iov/8a/asd+HMKiDmcoCo459RhXnNopIt4aXfrkdYUBYeWRY2VT71EWgjRAQm16BO6+tIjQwWvGaL/nnfI+ngT4UwPDTcLteGaUtwzqxLLxj7XeedfNxGOtR5pp5ZFjVVAfUQiLYTomOz6Fn1GV116ZKoKXjOIoZVB+T7UaJSoYQI2Gd8th28ejHRkpwNrngv12ggMbvo+qqLg1DKpsQqok0gLIZIkoRZ9SydfetQQ6QC6Wk4gGsGdnUNc19EjYRw/qMN9ZdNI1y/ohx4IE8rOaTosFFxaBnVWAXWRThueEOIoILu+hWiDQ1XIMQPoahnBWENdK04Yhu/YwWReW0rGIZGO7nTgu/M4zH1+fMcOoeKEYYl5Kgou3U1dpB+1EaXF5wghRHtki1qIVjgObEmrahnBWPTgDFUhfI8Xc8TBE8f40kHdgn6491UTcbrZPOM7id3tCuDSXdRHCqm1FJLe4R2Py93DhBCAhFqIFhyqgtfhR1XKCR0aaWzG9XucofmvJ6bEinW0mzIwasJUDR7K5hnfYe9pY4GGSLt1F/5IITUpRLonHy4ihEg/EmohDtF+pFcwOv+ZxJTK4FA2fnETF/3YxT9q9AN3Jju4Je3WXfijhdRYakqRnrZkAWagjmB2DlFPwxOuGh8uIg/PEOLoI6EW4oCOI/10YkplaCgv7X6U2CAvnJxD5WfVHFpjt+4kEC2kxtJI+pHSPfhwESFE+pL/2kXvE49T+Olmjn/7DQo/3QzxpFPYpvYj/etmkT6Rl0oeJRTztvpebs1BMNqvIdIpXIWVysNFhBBHj7QMtc/n4/bbb2f8+PGceeaZzJ07l7KyMgA2b97MlVdeyejRo5kyZQqrVq1q8trVq1czdepURo0axYwZM/jwww8T82KxGA888ABnnXUWo0ePZs6cOYn3BaisrGTu3LmMHTuW8ePHs2TJEqLRKCJ9DN64jqtvvJoZC67l4rt/yIwF13L1jVcf0fOmO470/yamNET6sbYjrRuEY/2osQxSfRBW4uEiZtsPF9EikU59uIgQIv2lZah/+MMfEggEeP311/nnP/+Jpmncdddd1NTUcMMNN3D55ZezadMmlixZwv3338+WLVsAePfdd7nvvvtYtmwZmzZt4tJLL2XOnDkEg0EAVqxYwfr163n++edZt24dTqeTxYsXJz735ptvxu12s27dOp577jneeecdVq5c2ROrQLSi8fhtQfEnWK4M6vILsVwZieO3hxPrxNndrUT6zILUIu3SDCKxftRETA7nzqCHPlykNZ39cBEhRO+QdqH++OOP2bx5M8uWLSM7O5vMzEzuu+8+FixYwGuvvYbX62XWrFnous7EiROZPn06zz7b8EjBVatWcfHFFzNmzBgMw2D27Nnk5OTwyiuvJOZff/31DBgwgMzMTBYtWsSbb77Jnj17KCkpYePGjdx22224XC4GDhzI3LlzE+8teliz47dRpwtUteH4bUF/zEA9E1YuT2k3eGJLWi1rNdJnFCQfaaemE4sX4LNcRA5zT3zjw0VctT6wm5X+wMNFqoYUdcrDRYQQvUfahXrLli0MHTqUv/zlL0ydOpVJkybxwAMPUFBQQHFxMcOGDWuy/NChQ9m2bRsAO3bsaHN+XV0d+/fvbzI/Pz8fj8fD9u3bKS4uxuv1UlhYmJh/4oknsm/fPmpra7vwG4tkdPbx2/Z2d59Z8JtWIt32MWlQscnHF3FjpXJQusXbdM/DRYQQvUvanfVdU1PD9u3bOfXUU1m9ejWhUIjbb7+dO+64g/z8fFyups8bdjqdBAIBAPx+f5vz/X4/AG63u8X8xnnNX9v4cyAQIDs7O+nvYKq98+5TjeNOx/Fn11Y1XFPsdbToNEDM4UCr85FdW0V1B+NvuC3owUgbicVtzij4LaPyVyaWrQqdyNrdjxKL5xyy3EEZug7kEowq2Hb8iNfd1xPO4R93PciZTzVcR+2q8xHXDSqKRrDpuvl8Pe4czCP6hJ6Xzn/O0pGsr9T1tXWWdqE2zYa/hhYtWoTD4SAzM5Obb76Zq666ihkzZhAKhZosHwqFyMjIABrC2tr8nJycRHQbj1c3f71t2y3mNf7c+P7JunFkbkrLp5u0HH/0BMhwkqnGwNVKqgIBcDu5YtwJcHpHx3BrAQvIOmSaDfwK+J9Dpg0j1/m/fHtYW+tDAbxALv81IsnvkYzTvwnfuww+/BAqKiA/n+zRoxnSx7ak0/LPWRqT9ZW6vrLO0i7UQ4cOJR6PE4lEcDgazn6NHzjuePLJJ/OHP/yhyfI7duygqKgIgKKiIoqLi1vMP+ecc/B4PBQWFjbZPV5eXo7P52PYsGHE43F8Ph8VFRXk5+cDsHPnTvr3709WVhapeHRr1ZHtAu0hpqpw48jc9By/OogZx5xAfvGn1Pfr33T3t22TWVZJRdEIXlAHwebWz4p2agoesw6oINxsd3fDlvTBSFeFTuTV3b8iFFOA6hbvpSsqhuYhGNW57uQu+neuD4H+Qxr+/9aWY+it0vrPWRqS9ZW63rLObu1wo6JB2oX6rLPOYuDAgdx5553cf//9hMNhHnroIS644AIuueQSli9fzsqVK5k1axbvv/8+a9as4fHHHwdg5syZzJs3j4suuogxY8bw7LPPUllZydSpUwGYMWMGK1asYOTIkeTk5LB06VLGjRvHoEGDABgzZgxLly7l3nvvpbq6mscff5yZM2em/B2suJ3Wfzg6kp7jV3hn9nymLVlAZtl+gtleombDXbtctT4sdybvzJ6PhUJrFy+7dIUM3UckXkU4Hjtkjs3YZpE+eEw6p8X7QEOkFTWbylAe1oFfItNznaU3WWepkfWVur6yztJuX5phGDzzzDNomsa0adOYNm0a/fv3Z+nSpeTk5PDUU0/x6quvMn78eBYvXszixYuZMGECABMnTuTuu+/mnnvuYdy4cbz88ss88cQTeL1eAObNm8e5557LrFmzOPfccwmHwzz88MOJz16+fDnRaJTzzz+fq666ismTJzN37tweWAuiNSXjJrN20YOUF43ADPrJqijFDPopLxrR7q013bpCjukjTmuRfoIxBc0j/Ui7kTa1bGqsfOoP5xosIYRIkWLbza8DEUfql5sre+VvcaaqcOvpeek//hSeLJWhK3jNKiJ2DZFWI/1UYsrBSLd+XEtTFJxaNjVWAXWRhvXTa9ZZGpF1lhpZX6nrLets4ej8pJZLu13fQnRIVSkdcXqHi2UaCh6jEiteQ9Q+9OLmlpGuCp3QbqR1RcWhZVFj5SciLYQQ3UFCLfqkLEPBY5YTjtW1EuknW0R6TcmjLSMdj5P/xee4/XXEc4ewa8AQ6mIIIUS3klCLPkUBsgzINssIxeqINTmy0xjp3yWmtBXpY7e8x+kvPI23sgzNUUD06xgjnNntPxO6+S75YadQ+PknSe2iF0KItkioRfpL8pi0AmQbNllmGaGYP4lIH9/q7u5jt7zHpBXLMOw4VuFJBEoV1JqvKNjzVZvPhB648U3GHrhJiRaJQDyOGo9hqxq2qhIzDKqGFLUfeiGEaIWEWqS1wRvXMWHlwQC2FTwVyDZtMo1SgtEAcZoeR2490o8SjDW7jjEe5/QXnsawYwQHnkZsb4x4RTnx9p4J/cYbXPCzBRj+OoLZOah6BM/+r1BjUeKaRk3/gcQNI/HwkPbOUBdCiOZkP5xIW8k+LUtVwOOIkWl83WqkxxQ8kVykgfwvPsdb/jVW/5OI7YsRLys/OLO1e4rH47BsGab/wMNCHE4yq8pR7DhR04FiQ2Z1+RE9PEQIcXSTUIv0lOTTsjQ7To4ZJUPfTyAabBnp/CcZm2SkAdz1tWgZhVjlGvGyihbzmz8Tut+2LbB9O0FPw8NC9FAQzQoT13RQFOKahhYOY4SCh/XwECGEkFCLtJTU07L272Hovk9x6fsJREM0v2hqTP6TjO33ZOLn6vCQdiOtohDPH0y0FNTdX7V81CQtnwnt8lWBZRE1G253awYDqPF44rW2oqLYNmq04ZalzUMvhBAdkVCLtOT2VaJFIokANhfzeNEH5pFt7cQfDScV6TW7Hmsz0grg0l3sHjCeSjMbV011Us+EDnpzwTRx1VSTu3snGRWlKPEYesRCt8KosSi2ohDXG04HaR56IYToiIRapKWAN4+YYaBb4RbzFLcb85gsbG8In97yfMjDibRbd+GPFFIb1djwnRuTfiZ02fDToKCArNKv0YNBbFXFVhrmKfE4WjRCXNeJOF2thl4IIToioRZpqXT4SKqGFOGq9TXZslWystAG5+CM76Xa6aTihGFNXndG/u9SijSAW3cSjPajJqJic7j3FLcbiq8oxHSt2Sy7zdALIURH5PIskZ5UlQ0HnpaVVd7wtKxYQQFmoQOHfwdWMMDma29sErwz8n/Hmf2eSPycVKQ1k2C0Hz5Lb/LQrZJxkykZe3aH12/327YFysupKzwGZ40PzQqj2DZx9WCs9YiFq9ZHedEIuY5aCJEyCbVIW41bthNWLievtgLNEyEeqaAq28vm2fPZe9rYxLJn5D/VJNL1Nf1Yv+FmgsfmtLnfyK0bhKOF1FgGsdZu353EPcUbTyYLegsIeHLRQ8ED10/rRB1O9FCQzKpy3v7eLXxw1XWyJS2ESJmEWqS1knGTKZswiRO+/gCz9gvqne6G3d2HBG90/lOc2e+3iZ9juzTUORbn1i7Fd+wQNs/4TpOoQ0OkrVg/aiImR/K0ysaTyXQrTMThIupyN11AVbEyMtl72pkSaSHEYZG/OURac+sKOc5aKobksevUM6gYOrxFpMcdEun4lyq1CwZQb/cj4nSTW7KDSSuWceyW9xLLuDSdSKwAn+UkcoT3HSkbfhqcdBLOGl9SZ4kLIUSqJNQibbl1Ba9ZTcyuxoq3fGxVa5GuWnQ8EX8mKCpR04k/twAjFOD0F56GeBynphOL98NnuY840kDDLw0LFxLJSO4scSGESJX87SHSUoaukGNWEbV9SUU6VqJRc/sxxKuNZksqhDOz8e7dxbFf7SZuF+Cz3J37MPkpU/j74lTPEhdCiOTIMWqRdjJ0Ba9ZScSuIdLKPbGbR7q+ph/qHItI3N1wiVQzUcPEoRuYtTq+cAbhzoz0AXvGncPOMzo+S1wIIVIloRZpJVNXyDYrseI1RO3WIv0/TSJdHR7M2xtu4ZyapehOi6jpbPEaw51J3O+inPwuiXRCEmeJCyFEquTXfZE2Mg0Fj1lJpN1I/ybxsy88iJdKHuOrY8fhO3YIjvpaaHYzUSXbg1mjUxX3sufEk7v6KwghRKeTUIu0kGWA1yxvZ0t6ZYtIryl5nEA0H1SVzTO+Q8TpJqOqHN0KgR3HcDhwV9pYZRZvz/ye7IYWQvRK8jeX6FEKkG3YeMwyQrHadiL968TPTSJ9wN7TxvLWnIVUDR6KEQqQEY1gWBlUaIW8euNP5IQuIUSvJceoRY9RgGzTJssoIxjzE2/lsZKtR/qxJpFutPe0sew99QwG7NmFq16jXOnP7uOHy5a0EKJXk1CLHqECHjNOhlFKMBok3uJBlTAq73/biHRBm+9r6gbVJ4zjS8tL8EhuOSaEEGlCQi26naZAthkjQy8lEA22kuiGSI8vXJH42Rce2HGkVQ2VXHxhD8FWb94thBC9j4RadCtNAa8ZwaWXEYiGUoj04+1G2qFqKOTisyTSQoi+RUItuo1+INIOrRR/NNzqMqPynj6sLWmFXKotDyGJtBCij5FQi25hqOA1wzi0MvxRq9VlGiL9eOLnmvBxByLdr833NVUNVcnBF5ZICyH6Jgm16HKmquA1gxhaGf5opNVlTm8l0i+WPN5hpDUlh2rLK7u7hRB9loRadCmHquA1A2hqOYF2Ij0hxUgbqoamePHJ2d1CiD5OQi26jENT8Jp+VKWcYCza6jKn5z2TcqR1RcVQsqm2cghIpIUQfZyEWnQJp6bgNetQlEpC7Ub6scTPyURaUxQcWhY+K08iLYQ4KkioRadriHQtKFXtRPr3KUdaRcGpZVBr5VMfkUgLIY4OEmrRqVy6gtf0YdvVhGOxVpdpiPSjiZ8bzu5uP9IK4NKd1EX6Udf6oW4hhOiT5CbIotO4dYUc00fcriYcTzLSVkOk/R1EOkN34o8WUmcprd4kRQgh+irZohadIkNX8JjVxGwfVhuRPi3v2ZaR3tVxpN26k0C0kBpLo+WztYQQom+TUIsjlqEreM1KInYtkXYiPbHwkcTPyUQawK07CEYL8Vk6cdmUFkIchSTU4ohk6goesxIrXtPqs6QBTss9zEhrByMt9zMRQhyt5Bi1OGyZhoLHUdFxpPs3j/RjHUY6QzcJxwrxWYZEWghxVJMtanFYsgzwmOWEY3VtRnpk7h+aRfrYA5EubPe93bpBONZPIi2EEEioxWHINiDbLCcUqyNmt17Skbl/4Kz+yxM/N0T68aQibcX64bMcyP1MhBBCQi1SoABZhn0g0vWdHmmXphOJFeCznETk9G4hhAAk1CJJCpBt2GSZZQRjfuIpRPqlJHZ3uzSdWLwfPstNRE7vFkKIBAm16JACZJs2WUYpwWiAeBu3HBmZ+8dWI10f7d/u+zsPibQlkRZCiCYk1KJdKuAx42QYpQSjwQ4i/avEz6lEOm4X4LPchCXSQgjRgoRatElVwGPEyTD2E4yGko50rXVMCpHOx2dlSKSFEKINEureJB6ncNtW3L5KAt48SoePBLVrLoVXFfCYMTL0UgLRYJv31x6Z+6cWkV6z6/EOI+1QNWw7D5+VSViuwRJCiDZJqHuJwRvXMWHlcnJ3FaNFIsQMg6ohRWyYPZ+ScZM79bM0BbKMKC69lEA01EGkH078XGsNYE1Jx1vSpqqhkEu1lSWRFkKIDsidyXqBwRvXMW3JAgqKP8FyZVCXX4jlyqCg+FOmLVnA4I3rOvXzPGakw0ifmvvnViL9OPWRAe2+t6lqqEoOPstDSCIthBAdklCnu3icCSuXYwbqqCsYQNTpAlUl6nRRV9AfM1DPhJXLIX7kFx7rCkAIl1aKv4NIn93/ocTPyUbaUDU0xYvP8hKUSAshRFIk1GmucNtWcncVE8zOAUVpOlNRCGZ7yd1VTOG2rUf0OboCHtMCvsYfDbW53OFHWsVQPPisHIJyyzEhhEiahDrNuX2VaJEIUdPR6vyo6UCLRHD7Kg/7MwwVchxhHNp+INLmcqfk/OWwIq0rjZHOJSCRFkKIlEio01zAm0fMMNCtcKvzdStMzDAIePMO6/1NVSHHDGFqpQSiVpvLnZLzFyYN+GXi58SJY0lE2lQ9+Kw8/BJpIYRImYQ6zZUOH0nVkCJctT5ofttO28ZV66NqSFHDpVopMlUFrxnA0MoIRNvfkj400nVW/wORPqbd99cVFVPLpkYiLYQQh01Cne5UlQ2z52O5M8kq348eCkI8jh4KklW+H8udyYbZ81O+ntqhKeSYAXQ19Ui/WPJ4h5HWFAWHlkWNlU+9RFoIIQ6bhLoXKBk3mbWLHqS8aARm0E9WRSlm0E950QjWLnow5euonZqC16xHVcsIxqJtLndKzqrDjrSzMdIRibQQQhwJueFJL1EybjIlY88+4juTuTQFr1mDrVQT6jDSv0j8nGykVUXBqWVQaxVQ1/aGuhBCiCRJqHsTVaV0xOmH/XK3ruA1fcTsaqxYrM3lTs5ZxcT+ByPtD+Tx4p7HqI91EGkUXJqbOqsftRJpIYToFLLr+yiRoSvkmFUNkY63HWl4tkmk4/tU7GvgnLse4Ngt77X5KgVw6S7qIoXURpQ2lxNCCJEaCfVRIENX8JqVRGxfu5E+Oec54N7Ez7FSjaofD8byZZFbsoNJK5a1GmsFcOsu/JFCai2lzTuaCSGESJ2Euo/LPBBpK15DpJ3bjJ6S8xwT+z+Y+DlWqlN9x2DiZQ6iphN/bgFGKMDpLzzd4nalbt1JMNqPmogqkRZCiE4moe7DMg0Fj6MCK15D1G4/0pMGHBppjeo7BhEvNQ9ZSiGcmY137y7yv/g8MdWtOQhG++GzdOSR0kII0fkk1H1UlqHgNcuxYrUpRZr9Kr6FA5tFukHUMFGjUZy11QC4dYNwrB8+y0CesSGEEF1Dzvrug7IM8JjlhGN17UZ6RLNIB4K5uG/RUMqAlp1Gj1jEdZ1Qdg5u3cCKFeKzTIm0EEJ0Idmi7mMaIx3qYEt6RM7zTD4k0nWRQl7+6rfgLMJRXwstjjbbOOpr8R07BH/RKURi/fBZDuSmY0II0bUk1H2EAmQb4DHLCMXqiDW/L/ghGiL934mf6yKFrNn1OHXR4+CGG4i63GRUlaNbIbDj6FaIjKpyIk43n/1/NxCjEJ/lInLkj8AWQgjRAQl1H9AQaZtss5RgrD7lSL+06zHqIsc2TJg4kbfnLqRq8FCMUICM6gqMUICqwUN596a72Xfy+fgibiw5c0wIIbqFHKPu5RQg27TJMkoJRgPE27lA6uScF5pEuj7Sj5d2PUZt5Lgmy309ciy7R5xB/hef46ytJpSdQ+3Qk1HVfHxWFmE5KC2EEN1GQt2LqYDHjJNhlBKMBjuM9DkDfp74uT7SjzW7Hm8R6YNvrlIxdDgAhqqhKzlUWx5CEmkhhOhWEupeSlXAY8bI0EsJRkPtR9q7OrVIH0JXVAzFQ7XlJShnjgkhRLeTUPdC2oFIu/VSAtFgu3cDO9m7mnOOeSDxc6qRNlUPPiuXgERaCCF6hIS6l9EU8JpRXHopgWioyyKtKQoOLQuflYdfIi2EED0mbc/6jsViXHPNNSxcuDAxbfPmzVx55ZWMHj2aKVOmsGrVqiavWb16NVOnTmXUqFHMmDGDDz/8sMn7PfDAA5x11lmMHj2aOXPmUFZWlphfWVnJ3LlzGTt2LOPHj2fJkiVEo20/r7knNEQ6gkvfj7/DSP/fYUe64ZnSmdRY+dRH2viUeJzCTzdz/NtvUPjp5hb3/xZCCNE50jbUjz76KO+9d/BJTTU1Ndxwww1cfvnlbNq0iSVLlnD//fezZcsWAN59913uu+8+li1bxqZNm7j00kuZM2cOwWAQgBUrVrB+/Xqef/551q1bh9PpZPHixYn3v/nmm3G73axbt47nnnuOd955h5UrV3brd26PrkCOGcGpleKPhttdtiHSyxI/N0S65dndbXFqmdRaBdS18UzpwRvXcfWNVzNjwbVcfPcPmbHgWq6+8WoGb1yX9PcRQgiRnLQM9TvvvMNrr73GhRdemJj22muv4fV6mTVrFrquM3HiRKZPn86zzz4LwKpVq7j44osZM2YMhmEwe/ZscnJyeOWVVxLzr7/+egYMGEBmZiaLFi3izTffZM+ePZSUlLBx40Zuu+02XC4XAwcOZO7cuYn37latbKnqCnhNC6e+n0As1UgXHIj0wA4/uuEp0i78kQLq2nim9OCN65i2ZAEFxZ9guTKoyy/EcmVQUPwp05YskFgLIUQnS7tj1JWVlSxatIjHH3+8yRZtcXExw4YNa7Ls0KFDee655wDYsWMHV1xxRYv527Zto66ujv379zd5fX5+Ph6Ph+3btwPg9XopLCxMzD/xxBPZt28ftbW1ZGdnp/QdTLX1yHVk4MY3OfOp5eTuKkaNRIgbBtXDR7LtxrlUn3YcgaiF0c5bD/P+H5MGHIy0P1LA30oeJxgd2O7roPF67AygP6FYHYbayi7veJyJK5fjCNRR128AKAoKEHO5qHc6ySzbz8SVy/l63CRQ0/J3wE7X+O/6cP+dH41knaVG1lfq+to6S6tQx+NxbrvtNr773e8yfPjwJvP8fj8ul6vJNKfTSSAQ6HC+3+8HwO12t5jfOK/5axt/DgQCKYf6xpG5KS0PwBtvwM9vh7o6yMsDhwM0jWylisF/uReOuREmTmznDf4CLDvk50IyjGe4cujgJAfgAAYABvPaGv/778O+L6CwgAyX0XJ+vzwy933BrfHdMHpMkp/bNxzWv/OjnKyz1Mj6Sl1fWWdpFerf/OY3mKbJNddc02Key+Wirq6uybRQKERGRkZifigUajE/JycnEd3G49XNX2/bdot5jT83vn8qHt1aldotNuNxZiy6j4LqmoYtVRQU3YU2KAdMH66dn1H90OO85jkJlJa/Iba+Jf0ItZFsoLrDj8/QnQRjHgLReuaemtvm+Adv/IJp/hD1bg8EWznRztbJDIRYu/ELSvQhyX//XsxUFW4c2fY6Ey3JOkuNrK/U9ZZ1duvpeUktl1ah/utf/0pZWRljx44FSIT373//O7fffjvr169vsvyOHTsoKioCoKioiOLi4hbzzznnHDweD4WFhezYsSOx+7u8vByfz8ewYcOIx+P4fD4qKirIz88HYOfOnfTv35+srKyUv4cVt1P6w1H46RZydhUTyM7BRgGHE21QDrbhw66uJJyZjWfvLjw7tifuFtZouPevLSL9Yslj1FqDkvrsDN2kLtKPaktDU+x2x1+bnUvMMNDCYaJOV4v5ejhMTDeozc5N6/84ukKq/86FrLNUyfpKXV9ZZ2l1IPHVV1/lgw8+4L333uO9997jkksu4ZJLLuG9995j6tSpVFRUsHLlSiKRCBs2bGDNmjWJ49IzZ85kzZo1bNiwgUgkwsqVK6msrGTq1KkAzJgxgxUrVrBnzx7q6+tZunQp48aNY9CgQQwZMoQxY8awdOlS6uvr2bNnD48//jgzZ87slu/t9lWiRSJETQcAiq6jOOPYvkoAooaJGo3irG26dTzc+1fOPeb+xM+pRtqtG4Rj/fBZRlLPlC4dPpKqIUW4an3Q/MEfto2r1kfVkCJKh49M6vOFEEJ0LK1C3Z6cnByeeuopXn31VcaPH8/ixYtZvHgxEyZMAGDixIncfffd3HPPPYwbN46XX36ZJ554Aq/XC8C8efM499xzmTVrFueeey7hcJiHH3448f7Lly8nGo1y/vnnc9VVVzF58mTmzp3bLd8t4M0jZhjo1iFndB8SQj1iEdd1Qtk5iWkneV9sGekvH8X8NMBxH7xD/o5t7V7b7NYNrFSfKa2qbJg9H8udSVb5fvRQsOGs9FCQrPL9WO5MNsyef9ScSCaEEN0hrXZ9N7ds2bImP48cOZI//elPbS5/2WWXcdlll7U6zzAMFixYwIIFC1qdn5+fz/Llyw9/sEegcUu1oPhT6hzOZnNtHPW1VA0eSsUJDbvtT/K+yH8cszSxhD9SwMa1c5j09C/w7t2FGo0S13V8xw5h84zvsPe0sU3e0aXpRGIF1FguIinuFioZN5m1ix5kwsqGs9NdtT5ihkF50Qg2zJ5PybjJh7UOhBBCtC6tQ33UOLClOm3JArLK9xMydbAz0a0QjvpaIk43m2d8B1SVk7xrOHdA0y3pja/9gNH//RRGyE8400PUMNEjFrklO5i0YhlvzVmYiLVT04nF++GzDv+Z0iXjJlMy9mwKt23F7ask4M1r2N0tW9JCCNHp5G/WNNG4pVpeNAIjGMDtq8IIBagaPDQR2oZIL0U5cNKXP1LAmi8fYej/rsUI+fHn9iNqOkFRiZpO/LkFGKEAp7/wNMTjODWduF1wRJFOUFVKR5zOl2dNoXTE6RJpIYToIrJFnUYat1SP2/kpA7Sv8RlWw+7uxJb0oZHOZ03JYxjbAnj37iKc6aHx3mIHKYQzs/Hu3cUxe0qoOH4iPiuDcB84C1IIIY4WEup0o6qUnzQS2zWAQNQH0EakH6fGGsRxte+gRqNEDbPVt4saJg5VxVmnUWNlEk7m9G4hhBBpQ/ZXprmTPC+1GWmAUHYOcV1Hj1itvt5wOLGN/pQr/QhJpIUQoteRUKexkzwvce4xS1rs7q455DrpihOG4Tt2CI76Wmj24EvFnYEj7KQyns/uE07uzqELIYToJBLqNHVC9gttRLrZvbtVlc0zvkPE6SajqhzdCoEdR9d1MgIa4VKbt785W072EkKIXkr+9k5DI3L+yFmFiw6JdF7rkT5g72ljeWvOQqoGD8UIBcgIhzDJplw5lld/8GO5tlkIIXoxOZkszQzNXsuFA+c3i/TjbUa60d7TxrL31DMoLPmCzJBKBcexa8hJsiUthBC9nIQ6zZye/0yTSL/UzpZ0c7qmUzf0TPZYefiTvi+oEEKIdCabW2nms+rLidsqtdZgXip5DJ81JKnX6YqKqWVTE5FICyFEXyJb1Gnm0+qZ7KqdTq7Thz9an9RrNEXBoWVSY+VTH5FICyFEXyJb1GkoZjuxk/wdSlMUnFomNVYBdRJpIYTocyTUvZiqKDi1DGqtftRFeno0QgghuoKEupdSUXBrbuqsftRKpIUQos+SUPdCKgou3UVtpJDaSPMHcQghhOhLJNS9jAK4dCf1kf7UWgpyVFoIIfo2CXUvogBu3YVfIi2EEEcNCXUv0RBpJ/5oITWWSrynBySEEKJbSKh7CbfuJBgtpMbSJNJCCHEUkVD3Am7NQTDaD5+lE5f93UIIcVSRUKc5t24QjhXiswxiEmkhhDjqSKjTVMMxaQNLIi2EEEc1CXUai8QK8FkO5BkbQghx9JKHcqSpUMxLraUTkYPSQghxVJNQp6Fw3CYa1ojZEmkhhDjaya7vNCXHpIUQQoCEWgghhEhrEmohhBAijUmohRBCiDQmoRZCCCHSmIRaCCGESGMSaiGEECKNSaiFEEKINCahFkIIIdKYhFoIIYRIYxJqIYQQIo1JqIUQQog0JqEWQggh0piEWgghhEhjEmohhBAijUmohRBCiDQmoRZCCCHSmN7TA+iLTFXp6SEclsZx99bx9wRZZ6mTdZYaWV+p62vrTLFt2+7pQQghhBCidbLrWwghhEhjEmohhBAijUmohRBCiDQmoRZCCCHSmIRaCCGESGMSaiGEECKNSaiFEEKINCahFkIIIdKYhFoAsG3bNr773e8ybtw4zj77bG6//Xaqqqp6elhpLxaLcc0117Bw4cKeHkra8/l83H777YwfP54zzzyTuXPnUlZW1tPDSmuffPIJs2bNYuzYsUyaNImf/exnWJbV08NKS1VVVUydOpV33303MW3z5s1ceeWVjB49milTprBq1aoeHOHhk1ALQqEQ//Vf/8Xo0aN56623eOmll/D5fNx55509PbS09+ijj/Lee+/19DB6hR/+8IcEAgFef/11/vnPf6JpGnfddVdPDyttxeNxvv/97zNt2jQ2btzIc889x1tvvcUTTzzR00NLO++//z5XX301u3fvTkyrqanhhhtu4PLLL2fTpk0sWbKE+++/ny1btvTgSA+PhFqwb98+hg8fzrx58zBNk5ycHK6++mo2bdrU00NLa++88w6vvfYaF154YU8PJe19/PHHbN68mWXLlpGdnU1mZib33XcfCxYs6Omhpa2amhrKy8uJx+M03ulZVVVcLlcPjyy9rF69mgULFnDLLbc0mf7aa6/h9XqZNWsWuq4zceJEpk+fzrPPPttDIz18EmrBCSecwJNPPommaYlpa9eu5ZRTTunBUaW3yspKFi1axC9+8Qv5izMJW7ZsYejQofzlL39h6tSpTJo0iQceeICCgoKeHlraysnJYfbs2TzwwAOMHDmSc889lyFDhjB79uyeHlpamTRpEq+//jrf+MY3mkwvLi5m2LBhTaYNHTqUbdu2defwOoWEWjRh2zYPPfQQ//znP1m0aFFPDyctxeNxbrvtNr773e8yfPjwnh5Or1BTU8P27dvZtWsXq1ev5v/+7/8oLS3ljjvu6Omhpa14PI7T6eSuu+7io48+4qWXXmLnzp0sX768p4eWVgoKCtD1lg+C9Pv9LX6JdjqdBAKB7hpap5FQi4T6+nrmz5/PmjVr+P3vf89JJ53U00NKS7/5zW8wTZNrrrmmp4fSa5imCcCiRYvIzMwkPz+fm2++mX//+9/4/f4eHl16ev3111m7di3f/va3MU2ToqIi5s2bxx//+MeeHlqv4HK5CIVCTaaFQiEyMjJ6aESHT55HLQDYvXs3119/PccccwzPPfccubm5PT2ktPXXv/6VsrIyxo4dC5D4y+Dvf/+7nFjWhqFDhxKPx4lEIjgcDqBhixFAnrTbuq+//rrFGd66rmMYRg+NqHcZNmwY69evbzJtx44dFBUV9dCIDp9sUQtqamq49tprOeOMM/jd734nke7Aq6++ygcffMB7773He++9xyWXXMIll1wikW7HWWedxcCBA7nzzjvx+/1UVVXx0EMPccEFF5CZmdnTw0tLkyZNory8nF//+tfEYjH27NnDihUrmD59ek8PrVeYOnUqFRUVrFy5kkgkwoYNG1izZg1XXHFFTw8tZRJqwQsvvMC+ffv429/+xpgxYxg9enTiHyE6g2EYPPPMM2iaxrRp05g2bRr9+/dn6dKlPT20tDV06FB+85vf8MYbbzB+/Hi+853vMGXKlBZnN4vW5eTk8NRTT/Hqq68yfvx4Fi9ezOLFi5kwYUJPDy1lii37nYQQQoi0JVvUQgghRBqTUAshhBBpTEIthBBCpDEJtRBCCJHGJNRCCCFEGpNQCyGEEGlMQi2EEEKkMQm1EEIIkcYk1EKkoXvuuYezzz6bysrKJtOj0ShXXXUV3//+97vtHtlTpkxh5MiRTe5Y1/hPT9w29eKLL+bFF1/s9s8VoqfIncmESEPhcJirrrqKwsJCfvvb3yamP/TQQ6xZs4YXXngBr9fbLWOZMmUKN954IzNmzOiWzxNCNCVb1EKkIYfDwUMPPcSmTZt45plnANi4cSMrV67k4Ycfpra2lh/84AeMHz+e8847j4ceeijxpCXbtvntb3/L9OnTGTt2LGeeeSY/+tGPEk/5WrhwIfPnz+eiiy5iwoQJ7N69mz/84Q9ccMEFjB07lunTp7Nq1aqkx1pSUsLo0aN59tlngYbHpU6dOpVf/OIXQEPoH330UaZNm8bo0aOZNWsWO3bsSLz+k08+4ZprruHMM8/kwgsvZOXKlYm9BY888gjXXXcdV1xxBePGjWPTpk1MmTKFF154AQDLsvjVr37F+eefz7hx47j++uspKSlJvPdJJ53EM888k/jsb33rW2zfvj0xf/369cycOZPRo0czZcoUfv/73yfmvf3228ycOZOxY8fKVrzoWbYQIm298MIL9umnn25/+umn9nnnnWf//ve/t/1+v33eeefZDz74oB0Khex9+/bZM2fOtB988EHbtm375Zdfts8++2z7yy+/tG3btnfs2GGPGzfO/stf/mLbtm3fcccd9qhRo+zt27fbNTU19u7du+1TTz3V3rlzp23btv3mm2/aI0eOtEtLS23btu3zzjvPfv7559sd5+rVq+1Ro0bZu3fvtm+99Vb729/+th2NRhOvnzRpkv3pp5/awWDQvuuuu+zzzz/ftizL3r9/vz1mzBj797//vW1Zll1cXGxPnTrV/uMf/2jbtm0vX77cHj58uP3222/b9fX1diQSaTKeZcuW2Zdffrm9e/duOxQK2Y888og9ZcoUOxQK2bZt28OGDbOvvvpqu6yszK6trbVnz55tX3fddbZt2/YXX3xhn3rqqfaqVavsSCRib9261R49erT95ptv2p999pl92mmn2WvXrrWj0aj9/vvv2+PHj7fffPPNzvpXK0TSZItaiDT2zW9+kwsvvJBvfetbia3Rf/3rX1iWxa233orD4WDAgAHcdNNNiS3ac845h+eee44hQ4ZQVVVFdXU1Xq+X0tLSxPuOGjWKYcOGkZ2djaZp2LbNn/70J95//30mTpzIRx99RL9+/RLL//SnP2Xs2LFN/jn0cYuXX345F1xwAddeey1vv/02v/zlL9E0LTH/e9/7HieffDJOp5Mf//jHfP3113zwwQe8+OKLnHjiicyaNQvDMBg6dCjf+973Et8FYODAgUycOJGMjAx0XU9MbxzzrbfeysCBA3E4HMybN49IJMK//vWvxHLXXHMNBQUFZGVlcdFFF7Fr1y4AXn75ZU455RRmzpyJruuceuqp/OEPf+CUU07hT3/6E+effz4XXnghmqZxxhlncNVVVzUZlxDdRe94ESFET7rxxhv561//yk033QTA3r17qaqq4swzz0wsY9s2kUiEyspKTNPkoYce4p///Ce5ubmcfPLJRCKRJiefHRrhY445hmeeeYYnn3ySH/zgB8RiMWbMmMFtt92Gw+EA4O677+7wGPU111zDiy++yOWXX05hYWGTeYMHD078f5fLhdfrpby8nL179/LJJ58wduzYxPx4PN4k8oeO9VBVVVUEAgFuuukmVPXgNkckEmHv3r2Jn/Pz8xP/X9f1xHooKyvjmGOOafKew4cPBxrW8YYNG5qMKxaLMWjQoHbXgRBdQUItRJprjFDj//bv359Bgwbx6quvJpapr6+nsrKS3Nxc7rnnHvbt28cbb7xBZmYmQJOtXwBFURL/v7KyklgsxmOPPUY8HueDDz5g/vz5HH/88cyaNSupMVqWxU9+8hMuueQS1q5dyze+8Q3OPffcxPxDt+b9fj/V1dUMGDCA/v37M378eH73u98l5ldXV+P3+1sd66FycnJwOBw89dRTjBo1KjH9iy++aPGLQmsGDBjAv//97ybTnn/+efLy8ujfvz/f/OY3uffeexPzysrKuu1MeyEOJbu+hehlzjvvPPx+P08++SSWZVFbW8sdd9zBLbfcgqIo1NfX43A40DSNcDjMU089xeeff04kEmn1/fbt28d1113HO++8g6qqicjl5OQkPaYHH3yQWCzG/fffz6233srChQspLy9PzP+f//kfSkpKCAaD3H///ZxwwgmMHj2a6dOn89FHH/Hiiy8SjUYpKyvjBz/4AcuWLevwM1VVZebMmfziF79g//79xONxVq9ezSWXXNLkhLK2XHzxxXz66af83//9H7FYjI8//phly5ah6zozZ87kpZde4q233iIej7Nr1y7+3//7fzz11FNJrxMhOotsUQvRy2RmZrJy5UqWLVvGk08+STweZ/z48axYsQKAm2++mR//+MecddZZuN1uxowZw2WXXcbnn3/e6vuNHDmSn/zkJ9xzzz2UlZWRlZXFt7/9bS666KLEMnfffTf33Xdfi9fOnTuXk046iT/84Q/85S9/wTRNrrnmGv7+97+zcOFCnnzySQDGjBnDvHnz2LdvH2eeeSa//e1vUVWVY489lieffJIHH3yQn/3sZ2iaxn/8x3+waNGipNbFHXfcwSOPPMK3v/1tfD4fAwcOZPny5YwYMaLD1w4aNIjf/va3/OIXv+C+++4jLy+PhQsXMmnSJAB++ctf8stf/pKbbroJl8vFJZdcwq233prUuIToTHIdtRCiS8l12EIcGdn1LYQQQqQxCbUQQgiRxmTXtxBCCJHGZItaCCGESGMSaiGEECKNSaiFEEKINCahFkIIIdKYhFoIIYRIYxJqIYQQIo1JqIUQQog0JqEWQggh0piEWgghhEhj/z8wLmfsgxeO9QAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 500x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "sns.lmplot(x=\"YearsExperience\",y=\"Salary\",data=df,scatter_kws={\"color\":'red'},line_kws={\"color\":'yellow'})\n",
    "sns.set_style('darkgrid')\n",
    "ax=plt.gca()\n",
    "plt.gca()\n",
    "plt.gca().set_facecolor('skyblue')                                                               \n",
    "                                                               "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "id": "c9fbecd5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours Studied</th>\n",
       "      <th>Previous Scores</th>\n",
       "      <th>Extracurricular Activities</th>\n",
       "      <th>Sleep Hours</th>\n",
       "      <th>Sample Question Papers Practiced</th>\n",
       "      <th>Performance Index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7</td>\n",
       "      <td>99</td>\n",
       "      <td>Yes</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>91.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4</td>\n",
       "      <td>82</td>\n",
       "      <td>No</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>65.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>51</td>\n",
       "      <td>Yes</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>45.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5</td>\n",
       "      <td>52</td>\n",
       "      <td>Yes</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>36.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7</td>\n",
       "      <td>75</td>\n",
       "      <td>No</td>\n",
       "      <td>8</td>\n",
       "      <td>5</td>\n",
       "      <td>66.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9995</th>\n",
       "      <td>1</td>\n",
       "      <td>49</td>\n",
       "      <td>Yes</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>23.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>7</td>\n",
       "      <td>64</td>\n",
       "      <td>Yes</td>\n",
       "      <td>8</td>\n",
       "      <td>5</td>\n",
       "      <td>58.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>6</td>\n",
       "      <td>83</td>\n",
       "      <td>Yes</td>\n",
       "      <td>8</td>\n",
       "      <td>5</td>\n",
       "      <td>74.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>9</td>\n",
       "      <td>97</td>\n",
       "      <td>Yes</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>95.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>7</td>\n",
       "      <td>74</td>\n",
       "      <td>No</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>64.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10000 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Hours Studied  Previous Scores Extracurricular Activities  Sleep Hours  \\\n",
       "0                 7               99                        Yes            9   \n",
       "1                 4               82                         No            4   \n",
       "2                 8               51                        Yes            7   \n",
       "3                 5               52                        Yes            5   \n",
       "4                 7               75                         No            8   \n",
       "...             ...              ...                        ...          ...   \n",
       "9995              1               49                        Yes            4   \n",
       "9996              7               64                        Yes            8   \n",
       "9997              6               83                        Yes            8   \n",
       "9998              9               97                        Yes            7   \n",
       "9999              7               74                         No            8   \n",
       "\n",
       "      Sample Question Papers Practiced  Performance Index  \n",
       "0                                    1               91.0  \n",
       "1                                    2               65.0  \n",
       "2                                    2               45.0  \n",
       "3                                    2               36.0  \n",
       "4                                    5               66.0  \n",
       "...                                ...                ...  \n",
       "9995                                 2               23.0  \n",
       "9996                                 5               58.0  \n",
       "9997                                 5               74.0  \n",
       "9998                                 0               95.0  \n",
       "9999                                 1               64.0  \n",
       "\n",
       "[10000 rows x 6 columns]"
      ]
     },
     "execution_count": 154,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import  pandas as pd\n",
    "df=pd.read_csv(r\"C:\\Users\\GATEWAY\\Desktop\\uber\\LR_Student_Performance.csv\")\n",
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "add45769",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 6)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "cf05bc56",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 10000 entries, 0 to 9999\n",
      "Data columns (total 6 columns):\n",
      " #   Column                            Non-Null Count  Dtype  \n",
      "---  ------                            --------------  -----  \n",
      " 0   Hours Studied                     10000 non-null  int64  \n",
      " 1   Previous Scores                   10000 non-null  int64  \n",
      " 2   Extracurricular Activities        10000 non-null  object \n",
      " 3   Sleep Hours                       10000 non-null  int64  \n",
      " 4   Sample Question Papers Practiced  10000 non-null  int64  \n",
      " 5   Performance Index                 10000 non-null  float64\n",
      "dtypes: float64(1), int64(4), object(1)\n",
      "memory usage: 468.9+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4ad5e67b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours Studied</th>\n",
       "      <th>Previous Scores</th>\n",
       "      <th>Sleep Hours</th>\n",
       "      <th>Sample Question Papers Practiced</th>\n",
       "      <th>Performance Index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>4.992900</td>\n",
       "      <td>69.445700</td>\n",
       "      <td>6.530600</td>\n",
       "      <td>4.583300</td>\n",
       "      <td>55.224800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>2.589309</td>\n",
       "      <td>17.343152</td>\n",
       "      <td>1.695863</td>\n",
       "      <td>2.867348</td>\n",
       "      <td>19.212558</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>40.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>10.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>3.000000</td>\n",
       "      <td>54.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>40.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.000000</td>\n",
       "      <td>69.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>55.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>7.000000</td>\n",
       "      <td>85.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>7.000000</td>\n",
       "      <td>71.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>9.000000</td>\n",
       "      <td>99.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>100.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       Hours Studied  Previous Scores   Sleep Hours  \\\n",
       "count   10000.000000     10000.000000  10000.000000   \n",
       "mean        4.992900        69.445700      6.530600   \n",
       "std         2.589309        17.343152      1.695863   \n",
       "min         1.000000        40.000000      4.000000   \n",
       "25%         3.000000        54.000000      5.000000   \n",
       "50%         5.000000        69.000000      7.000000   \n",
       "75%         7.000000        85.000000      8.000000   \n",
       "max         9.000000        99.000000      9.000000   \n",
       "\n",
       "       Sample Question Papers Practiced  Performance Index  \n",
       "count                      10000.000000       10000.000000  \n",
       "mean                           4.583300          55.224800  \n",
       "std                            2.867348          19.212558  \n",
       "min                            0.000000          10.000000  \n",
       "25%                            2.000000          40.000000  \n",
       "50%                            5.000000          55.000000  \n",
       "75%                            7.000000          71.000000  \n",
       "max                            9.000000         100.000000  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f56ccd86",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours Studied</th>\n",
       "      <th>Previous Scores</th>\n",
       "      <th>Extracurricular Activities</th>\n",
       "      <th>Sleep Hours</th>\n",
       "      <th>Sample Question Papers Practiced</th>\n",
       "      <th>Performance Index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7</td>\n",
       "      <td>99</td>\n",
       "      <td>Yes</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>91.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4</td>\n",
       "      <td>82</td>\n",
       "      <td>No</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>65.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>51</td>\n",
       "      <td>Yes</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>45.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5</td>\n",
       "      <td>52</td>\n",
       "      <td>Yes</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>36.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7</td>\n",
       "      <td>75</td>\n",
       "      <td>No</td>\n",
       "      <td>8</td>\n",
       "      <td>5</td>\n",
       "      <td>66.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours Studied  Previous Scores Extracurricular Activities  Sleep Hours  \\\n",
       "0              7               99                        Yes            9   \n",
       "1              4               82                         No            4   \n",
       "2              8               51                        Yes            7   \n",
       "3              5               52                        Yes            5   \n",
       "4              7               75                         No            8   \n",
       "\n",
       "   Sample Question Papers Practiced  Performance Index  \n",
       "0                                 1               91.0  \n",
       "1                                 2               65.0  \n",
       "2                                 2               45.0  \n",
       "3                                 2               36.0  \n",
       "4                                 5               66.0  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "c1d9687c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours Studied</th>\n",
       "      <th>Previous Scores</th>\n",
       "      <th>Extracurricular Activities</th>\n",
       "      <th>Sleep Hours</th>\n",
       "      <th>Sample Question Papers Practiced</th>\n",
       "      <th>Performance Index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>9995</th>\n",
       "      <td>1</td>\n",
       "      <td>49</td>\n",
       "      <td>Yes</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>23.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>7</td>\n",
       "      <td>64</td>\n",
       "      <td>Yes</td>\n",
       "      <td>8</td>\n",
       "      <td>5</td>\n",
       "      <td>58.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>6</td>\n",
       "      <td>83</td>\n",
       "      <td>Yes</td>\n",
       "      <td>8</td>\n",
       "      <td>5</td>\n",
       "      <td>74.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>9</td>\n",
       "      <td>97</td>\n",
       "      <td>Yes</td>\n",
       "      <td>7</td>\n",
       "      <td>0</td>\n",
       "      <td>95.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>7</td>\n",
       "      <td>74</td>\n",
       "      <td>No</td>\n",
       "      <td>8</td>\n",
       "      <td>1</td>\n",
       "      <td>64.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Hours Studied  Previous Scores Extracurricular Activities  Sleep Hours  \\\n",
       "9995              1               49                        Yes            4   \n",
       "9996              7               64                        Yes            8   \n",
       "9997              6               83                        Yes            8   \n",
       "9998              9               97                        Yes            7   \n",
       "9999              7               74                         No            8   \n",
       "\n",
       "      Sample Question Papers Practiced  Performance Index  \n",
       "9995                                 2               23.0  \n",
       "9996                                 5               58.0  \n",
       "9997                                 5               74.0  \n",
       "9998                                 0               95.0  \n",
       "9999                                 1               64.0  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0830a27b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours Studied</th>\n",
       "      <th>Previous Scores</th>\n",
       "      <th>Extracurricular Activities</th>\n",
       "      <th>Sleep Hours</th>\n",
       "      <th>Sample Question Papers Practiced</th>\n",
       "      <th>Performance Index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9995</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9996</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9997</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9998</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9999</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10000 rows × 6 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Hours Studied  Previous Scores  Extracurricular Activities  Sleep Hours  \\\n",
       "0             False            False                       False        False   \n",
       "1             False            False                       False        False   \n",
       "2             False            False                       False        False   \n",
       "3             False            False                       False        False   \n",
       "4             False            False                       False        False   \n",
       "...             ...              ...                         ...          ...   \n",
       "9995          False            False                       False        False   \n",
       "9996          False            False                       False        False   \n",
       "9997          False            False                       False        False   \n",
       "9998          False            False                       False        False   \n",
       "9999          False            False                       False        False   \n",
       "\n",
       "      Sample Question Papers Practiced  Performance Index  \n",
       "0                                False              False  \n",
       "1                                False              False  \n",
       "2                                False              False  \n",
       "3                                False              False  \n",
       "4                                False              False  \n",
       "...                                ...                ...  \n",
       "9995                             False              False  \n",
       "9996                             False              False  \n",
       "9997                             False              False  \n",
       "9998                             False              False  \n",
       "9999                             False              False  \n",
       "\n",
       "[10000 rows x 6 columns]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "4d881bb6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Hours Studied                                                                     642\n",
       "Previous Scores                                                                  8865\n",
       "Extracurricular Activities          NoYesNoYesYesNoNoNoNoNoYesNoYesNoNoNoYesYesYes...\n",
       "Sleep Hours                                                                       819\n",
       "Sample Question Papers Practiced                                                  585\n",
       "Performance Index                                                              7094.0\n",
       "dtype: object"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#to check duplicate values\n",
    "duplicate_rows=df.duplicated()\n",
    "df[duplicate_rows].sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "f222c338",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "127"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "duplicate_rows.sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "6b4d6b93",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Before dropping duplicates: (10000, 6)\n",
      "After dropping duplicate: (9873, 6)\n"
     ]
    }
   ],
   "source": [
    "print(\"Before dropping duplicates:\",df.shape)\n",
    "df.drop_duplicates(inplace=True)\n",
    "print(\"After dropping duplicate:\",df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "4ea079d6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('float64')"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#based on index value try to check the performance\n",
    "response=df['Performance Index']\n",
    "response.dtype\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "36e03c28",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Performance Index')"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.plot(response.index,response)\n",
    "plt.xlabel('index')\n",
    "plt.ylabel(\"Performance Index\",color='green')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "f6376e22",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.JointGrid at 0x223652bbc10>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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dtll9KlPfWqN7mhtOX78OAGghx9+mIiItNY0NjZGX/jCF+LPlEol6uvro7/7u78jIqLp6WlyXZfuv//++DOHDx8mKSX9x3/8BxERPffccwSAfvKTn8SfeeyxxwgAPf/886vxaGeFubk5uuSSS+ihhx6im2++ORZSXG+t+eQnP0k33nhj299zvbXmtttuo9///d+v+9lv//Zv0/vf/34i4nprRaOQWs06+u53v0tSSjp8+HD8mW9+85uUyWRoZmbmrDwvYwdv7a0QlUoFu3fvxlvf+ta6n7/1rW/Fo48+ukZ3tbbMzMwAAAYHBwEA+/btw+TkZF0dZTIZ3HzzzXEd7d69G77v131mfHwcO3fujD/z2GOPoa+vD69//evjz/zar/0a+vr61nVdf+QjH8Ftt92Gt7zlLXU/53przXe+8x1cd911ePe7342RkRFce+21+MpXvhL/nuutNTfeeCP+8z//Ey+++CIA4Be/+AV+/OMf4+1vfzsArrckrGYdPfbYY9i5cyfGx8fjz9x6660ol8t129jM2sGmxSvEyZMnEYYhRkdH634+OjqKycnJNbqrtYOI8PGPfxw33ngjdu7cCQBxPbSqowMHDsSf8TwPAwMDTZ+pXj85OYmRkZGmMkdGRtZtXd9///3Ys2cPfvaznzX9juutNXv37sWXvvQlfPzjH8ef/dmf4fHHH8fHPvYxZDIZfPCDH+R6a8MnP/lJzMzM4LLLLoNSCmEY4p577sF73/teANzekrCadTQ5OdlUzsDAADzPW/f1eL7AQmqFEULU/T8RNf3sQuDOO+/EU089hR//+MdNv0tTR42fafX59VrXBw8exF133YUHH3wQ2Wy27ee43urRWuO6667D5z//eQDAtddei2effRZf+tKX8MEPfjD+HNdbPf/0T/+Er3/96/jGN76BK6+8Ek8++STuvvtujI+P44477og/x/XWndWqo/O9Htc7vLW3QgwPD0Mp1bRCOH78eNNq4nznox/9KL7zne/gBz/4ASYmJuKfj42NAUDHOhobG0OlUsHU1FTHzxw7dqyp3BMnTqzLut69ezeOHz+OXbt2wXEcOI6Dhx9+GH/zN38Dx3HiZ+J6q2fjxo244oor6n52+eWX45VXXgHA7a0df/Inf4JPfepTeM973oOrrroKH/jAB/DHf/zHuPfeewFwvSVhNetobGysqZypqSn4vr/u6/F8gYXUCuF5Hnbt2oWHHnqo7ucPPfQQbrjhhjW6q9WFiHDnnXfigQcewPe//31s37697vfbt2/H2NhYXR1VKhU8/PDDcR3t2rULruvWfebo0aN45pln4s9cf/31mJmZweOPPx5/5qc//SlmZmbWZV2/+c1vxtNPP40nn3wy/nfdddfhfe97H5588kns2LGD660Fb3jDG5rSa7z44ovYunUrAG5v7VhYWICU9UO/UipOf8D11p3VrKPrr78ezzzzDI4ePRp/5sEHH0Qmk8GuXbvO6nMyCVnlw+3nNdX0B1/96lfpueeeo7vvvpsKhQLt379/rW9tVfjDP/xD6uvrox/+8Id09OjR+N/CwkL8mS984QvU19dHDzzwAD399NP03ve+t+VXhicmJuh73/se7dmzh37913+95VeGr776anrsscfoscceo6uuumrdfK06CbXf2iPiemvF448/To7j0D333EO/+tWv6B//8R8pn8/T17/+9fgzXG/N3HHHHbRp06Y4/cEDDzxAw8PD9IlPfCL+DNeb+RbtE088QU888QQBoL/6q7+iJ554Ik5ns1p1VE1/8OY3v5n27NlD3/ve92hiYoLTH5xDsJBaYe677z7aunUreZ5Hr3nNa+Kv/l8IAGj572tf+1r8Ga01feYzn6GxsTHKZDJ000030dNPP133dxYXF+nOO++kwcFByuVy9I53vINeeeWVus+cOnWK3ve+91GxWKRisUjve9/7aGpqahWecnVoFFJcb635t3/7N9q5cydlMhm67LLL6Mtf/nLd77nempmdnaW77rqLtmzZQtlslnbs2EGf/vSnqVwux5/heiP6wQ9+0HI8u+OOO4hodevowIEDdNttt1Eul6PBwUG68847qVQqnc3HZywQRERrEwtjGIZhGIZZ3/AZKYZhGIZhmJSwkGIYhmEYhkkJCymGYRiGYZiUsJBiGIZhGIZJCQsphmEYhmGYlLCQYhiGYRiGSQkLKYZhGIZhmJSwkGIYhmEYhkkJCymGYRiGYZiUsJBiGIZhGIZJCQsphmEYhmGYlLCQYhiGYRiGScn/D/JG4P0nR9U7AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 600x600 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "sns.jointplot(x=response.index,y=\"Performance Index\",data=df,kind='hex',color='violet')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "4342567a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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jzjJ2pyF63BijaTP5+PWW2/lhhtuwOPxmFxdw2Nqq3tJbeXl5Tz88MP07duX119/HQODUGGIw70OU3VqlcKJiDQo0UZRjpxzhCPdj1CdU43f7+exxx7jiiuuYMGCBcRiMbNLTFsKKAIktgw///zz9O7dm+eee47q6moiTSKUnV9GxZkVxL0640JEGqijpyeXnV9G+c/KibljHDhwgN///vcMGjSI999/3+wK05IGUNOcYRgsXbqURx55hOLiYgCiWVGqulQRaRZRLxMRSR82CLcMEy4I493lxbvVy6ZNm7jpppu48MILuffee+nQoYPZVaYNBZQ0tm3bNh566KHkq4NYZgx/Z/UyEZE054BA+wDBVkF8WxM9VN577z0++OADrr/+esaOHUtWVpbZVTZ4ug2loaqqKh5++GH69+/P+++/j2E38Hf0U9qrlFCRwomICCS60ladUpU4sqN5mGg0ynPPPUffvn157bXX1Dq/julWlEYMw2DRokX07duX5557jmg0Srh5mNILS/F39mM49csmIvJVMV+M8m7lHOl2hKg3ysGDB7n77rsZNWoUn332mdnlNVgKKGli37593Hzzzfzud7/j4MGDRL1RjnQ7Qnm3cmI+rVIXEfk+keYRSi8spapzFYbDYN26dQwePJjHHnuMSCRidnkNjgJKA2cYBi+//DL9+vVj5cqVGHaDqs5VlF5YSqS5fqFERE6IAwIdAxy+6DChghCxWIypU6cyePBg/vOf/5hdXYOigNKA1Yya/PGPf6SqqorqRtWUXlBKoGNA7elFRH6CuCdOxdkVlJ9VTtwV5/PPP2fo0KEaTalFCigN1JtvvnncqElll0rKzisjlqXpHBGR2hJuEebwhYcJtTh+NGXHjh1ml5byFFAamFgsxp///GfGjx9/3KhJsH1QPU1EROqAkWlQcdbxoylDhgxh+fLlZpeW0hRQGpAjR45w66238swzzwDgb++n7FyNmoiI1IdwizClF5QSyYtQWVnJmDFjmDJlirYj/0gKKA3Eli1buOaaa5JTOuVnluPv4td3WESkHsXdcY70OEKgdQDDMHj88ce54447CAQCZpeWctRJtgHYvHkz119/PUeOHCHmiVHetZxoTtTsskRE0pMdqk6rIpobJXtjNkuWLOHIkSNMmzZNJySfAL2+TnHbtm3jxhtv5MiRI1TnVlN6fqnC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YicnJ4devXqZXJ2I1DcFFBGxhG7dupGTk4M9Ysd5xImrJLFg9qKLLiIjo5ZOQBaRlKGAIiKWkJGRwUUXXQQkRlEySzIBuOyyy8wsS0RMooAiIpbRo0cPAFyHXTiqEq1kzznnHDNLEhGTOL//ISIi9aNLly4AZBxJTOk0a9aMpk2bmlmSiJhEIygiYhmdO3c+ro39ySefbGI1ImImBRQRsQy3201RUVHyzx07djSxGhExkwKKiFhKfn5+8v2CggITKxERMymgiIilfDmgNG/e3MRKRMRMCigiYilfDiUKKCLpSwFFRCwlNzf3G98XkfSigCIiluLz+ZLvZ2VlmViJiJhJAUVELMXlciXfV0ARSV9q1CYilnLuueeSl5dH586djxtNEZH0ooAiIpbSunVrVq1ahc1mw2azmV2OiJhEAUVELOfL3WRFJD3pWUBEREQsRwFFRERELEcBRURERCxHAUVEREQsRwFFRERELMfUgPLUU0/Rrl073G43Xbt25b333jOzHBEREbEI0wLKvHnzGD9+PPfddx/r16/nwgsv5Morr2T37t1mlSQiIiIWYTMMwzDjL+7Rowdnn302U6ZMSV47+eSTGThwIBMnTvzOz62oqCA3N5fy8nJycnLqulQRERGpBSdy/zZlBCUSibBu3Tr69Olz3PU+ffqwatWqrz0+HA5TUVFx3JuIiIg0XKYElEOHDhGLxcjPzz/uen5+Pvv37//a4ydOnEhubm7yraioqL5KFREREROYukj2q+dsGIbxjWdv3HvvvZSXlyffiouL66tEERERMYEpZ/E0bdoUh8PxtdGSkpKSr42qAGRmZpKZmVlf5YmIiIjJTAkoLpeLrl27snTpUn7xi18kry9dupQBAwZ87+fXrOvVWhQREZHUUXPf/iH7c0w7zfiuu+7iuuuuo1u3bpx33nlMmzaN3bt3M2bMmO/93MrKSgCtRREREUlBlZWV5ObmfudjTAsow4YN4/Dhwzz44IPs27eP0047jcWLF9OmTZvv/dzCwkKKi4vJzs7+xjUrIpK6KioqKCoqori4WG0ERBoYwzCorKyksLDwex9rWh8UEZFvoj5HIgI6i0dEREQsSAFFRERELEcBRUQsJTMzk//5n/9RawGRNKc1KCIiImI5GkERERERy1FAEREREctRQBERERHLUUARERERy1FAERFLeeqpp2jXrh1ut5uuXbvy3nvvmV2SiJhAAUVELGPevHmMHz+e++67j/Xr13PhhRdy5ZVXsnv3brNLE5F6pm3GImIZPXr04Oyzz2bKlCnJayeffDIDBw5k4sSJJlYmIvVNIygiYgmRSIR169bRp0+f46736dOHVatWmVSViJhFAUVELOHQoUPEYjHy8/OPu56fn8/+/ftNqkpEzKKAIiKWYrPZjvuzYRhfuyYiDZ8CiohYQtOmTXE4HF8bLSkpKfnaqIqINHwKKCJiCS6Xi65du7J06dLjri9dupSePXuaVJWImMVpdgEiIjXuuusurrvuOrp168Z5553HtGnT2L17N2PGjDG7NBGpZwooImIZw4YN4/Dhwzz44IPs27eP0047jcWLF9OmTRuzSxOReqY+KCIiImI5WoMiIiIilqOAIiIiIpajgCIiIiKWo4AiIiIilqOAIiIiIpajgCIiIiKWo4AiIiIilqOAIiIiIpajgCIiIiKWo4AiIiIilqOAIiIiIpajgCIiIiKW8/8BkGyhxyuKR+YAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.violinplot(response,color='green')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "id": "b84cc771",
   "metadata": {},
   "outputs": [],
   "source": [
    "#find min and max values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "125f9119",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "100.0"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ma=df['Performance Index'].max()\n",
    "ma\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "19567457",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10.0"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mi=df['Performance Index'].min()\n",
    "mi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "00499df8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(df['Performance Index']==mi).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "94149f3f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(df['Performance Index']==ma).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "a8d26cb8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "49287"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Hours Studied'].sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "b28fa7e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "9"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Hours Studied'].max()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "012430df",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Hours Studied'].min()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "80955c98",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Hours Studied</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1133</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1122</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>1118</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1110</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>1099</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1077</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1074</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1071</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>1069</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               count\n",
       "Hours Studied       \n",
       "1               1133\n",
       "6               1122\n",
       "7               1118\n",
       "3               1110\n",
       "9               1099\n",
       "2               1077\n",
       "8               1074\n",
       "4               1071\n",
       "5               1069"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.DataFrame(df['Hours Studied'].value_counts())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "b6d4dc6d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([7, 4, 8, 5, 3, 6, 2, 1, 9], dtype=int64)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Hours Studied'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "f5ac7b75",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Hours Studied', ylabel='Count'>"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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9fVVVFSUfv3rXysiIkJBQUF6++239fnnnys8PLxxh2nU41GolVr4KbOZM2daXbt2tY4dO+bqUZyuoqLCOnDggJWbm2s99dRTVseOHa3du3e7eiynOHr0qBUYGGjl5+fbt7WkU2bnKisrs4KCgqwlS5a4ehSn8fDwsKKiohy2xcbGWjfeeKOLJmo40dHR1ujRo109hlO9/fbbVufOna23337b+uqrr6w//OEPlr+/v7Vq1SpXj+ZUBw8etG655RZLkuXm5mYNGDDAmjRpkhUREdGoc3CECA0qNjZW77//vrZs2aLOnTu7ehyna926tf2i6v79+ys3N1e/+93v9Nprr7l4ssuXl5enwsJC9evXz76tqqpKW7ZsUVpamioqKlrURcg+Pj7q06ePDhw44OpRnKZTp041jlhGREToz3/+s4smahhHjhzRxo0btXbtWleP4lRPPPGEnnrqKd13332SpD59+ujIkSNKSkrSlClTXDyd81x11VXKzs7WqVOnVFJSok6dOunee+9VWFhYo85BEKFBWJal2NhYZWZmavPmzY3+F9tVLMtSRUWFq8dwiuHDh2vnzp0O2x566CH17NlTTz75ZIuKIUmqqKjQ3r17dfPNN7t6FKcZPHhwjY+72L9/v8u/1NrZVq5cqcDAQN15552uHsWpTp8+rVatHC/1dXNza3Fvu/+Zj4+PfHx8VFRUpPXr12vx4sWN+vMJIhcpKyvTwYMH7fcPHz6s/Px8+fv7q0uXLi6czDliYmK0Zs0a/eUvf5Gvr68KCgokSX5+fvL29nbxdM7x9NNP6/bbb1doaKhKS0uVkZGhzZs3a926da4ezSl8fX1rXPPl4+OjgICAFnEt2Lx58zRmzBh16dJFhYWFevHFF1VSUtKifvN+/PHHNWjQICUmJmrChAnatm2bli9fruXLl7t6NKeprq7WypUrNWXKFLm7t6x/0saMGaNFixapS5cu6t27t7744gslJydr2rRprh7NqdavXy/LstSjRw8dPHhQTzzxhHr06KGHHnqocQdp1BN0sPv4448tSTVuU6ZMcfVoTlHbvkmyVq5c6erRnGbatGlW165drdatW1tXXHGFNXz4cGvDhg2uHqtBtaRriO69916rU6dOloeHhxUSEmKNHz++xVz/9Uv/+7//a0VGRlqenp5Wz549reXLl7t6JKdav369Jcnat2+fq0dxupKSEmv27NlWly5dLC8vL6t79+5WfHy8VVFR4erRnOqdd96xunfvbrVu3doKDg62YmJirB9++KHR57BZlmU1boIBAAA0LXwOEQAAMB5BBAAAjEcQAQAA4xFEAADAeAQRAAAwHkEEAACMRxABAADjEUQAAMB4BBEANJJu3bopJSXFft9ms+m99967rNecOnWq7rrrrst6DQAEEQAnqesf5s2bN8tms+mHH35o9Jku5NChQ7r//vsVEhIiLy8vde7cWePGjdP+/fslSV9//bVsNpvy8/Mb5OefPHlSt99+e4O8NoBL07K+CQ+Asc6cOSMPD4+LXl9ZWamRI0eqZ8+eWrt2rTp16qTjx4/rgw8+UHFxcQNO+m/BwcGN8nMAXBhHiAA0uj//+c/q3bu3PD091a1bNy1ZssTh8dpOJbVv316rVq2S9O8jN3/60580dOhQeXl5KT09XUeOHNGYMWPUoUMH+fj4qHfv3vrggw9qnWHPnj06dOiQXnnlFd14443q2rWrBg8erEWLFmnAgAGSpLCwMElS3759ZbPZNHToUEnS0KFDFRcX5/B6d911l6ZOnWq/X1hYqDFjxsjb21thYWH64x//WGOGc/fzxIkTuvfee9WhQwcFBARo3Lhx+vrrr+2PV1VVac6cOWrfvr0CAgI0f/588XWUgHMQRAAaVV5eniZMmKD77rtPO3fuVEJCgp599ll77FyKJ598UrNmzdLevXs1atQoxcTEqKKiQlu2bNHOnTv10ksvqW3btrU+94orrlCrVq307rvvqqqqqtY127ZtkyRt3LhRJ0+e1Nq1ay96tqlTp+rrr7/WRx99pHfffVevvPKKCgsL61x/+vRpDRs2TG3bttWWLVuUk5Ojtm3b6rbbblNlZaUkacmSJXrzzTe1YsUK5eTk6Pvvv1dmZuZFzwSgbpwyA+A0f/3rX2sEyLmxkZycrOHDh+vZZ5+VJF1zzTXas2ePfvvb3zocYbkYcXFxGj9+vP3+0aNHdffdd6tPnz6SpO7du9f53CuvvFIvv/yy5s+frxdeeEH9+/fXsGHDNGnSJPvzrrjiCklSQEDAJZ3e2r9/vz788EN99tlnGjhwoCRpxYoVioiIqPM5GRkZatWqld544w3ZbDZJ0sqVK9W+fXtt3rxZ0dHRSklJ0YIFC3T33XdLkl599VWtX7/+oucCUDeOEAFwmmHDhik/P9/h9sYbbzis2bt3rwYPHuywbfDgwTpw4ECdR2rq0r9/f4f7s2bN0osvvqjBgwfr+eef11dffXXe58fExKigoEDp6emKiorS//zP/6h3797Kysq6pDnOtXfvXrm7uzvM17NnT7Vv377O5+Tl5engwYPy9fVV27Zt1bZtW/n7++vHH3/UP/7xDxUXF+vkyZOKioqyP+fcnwGg/ggiAE7j4+Ojq6++2uF25ZVXOqyxLMt+BOSX237JZrPV2HbmzJlaf94vPfzwwzp06JAmT56snTt3qn///kpNTT3vzL6+vho7dqwWLVqkL7/8UjfffLNefPHF8z6nVatW553v58fO3c/zqa6uVr9+/WoE5f79+zVx4sSLfh0A9UMQAWhUvXr1Uk5OjsO2rVu36pprrpGbm5ukn05VnTx50v74gQMHdPr06Yt6/dDQUD366KNau3at5s6dq9dff/2iZ7PZbOrZs6dOnTolSWrdurWkmqf9zp2vqqpKu3btst+PiIjQ2bNntX37dvu2ffv2nfejB2644QYdOHBAgYGBNaLSz89Pfn5+6tSpkz777DP7c86ePau8vLyL3j8AdSOIADSquXPnatOmTfr1r3+t/fv3a/Xq1UpLS9O8efPsa2699ValpaVpx44d2r59ux599NGLekt9XFyc1q9fr8OHD2vHjh366KOP6rxuJz8/X+PGjdO7776rPXv26ODBg1qxYoXefPNNjRs3TpIUGBgob29vrVu3Tt9++6397fi33nqr/va3v+lvf/ub/v73v2vmzJkOsdOjRw/ddttteuSRR/T5558rLy9PDz/8sLy9veucfdKkSerYsaPGjRunTz75RIcPH1Z2drZmz56t48ePS5Jmz56t3/zmN8rMzKz15wKoP4IIQKO64YYb9Kc//UkZGRmKjIzUc889p4ULFzpcUL1kyRKFhobqlltu0cSJEzVv3jy1adPmgq9dVVWlmJgYRURE6LbbblOPHj30yiuv1Lq2c+fO6tatm1544QUNHDhQN9xwg373u9/phRdeUHx8vKSfrtF5+eWX9dprrykkJMQeStOmTdOUKVP04IMPasiQIQoLC9OwYcMcXn/lypUKDQ3VkCFDNH78eP3qV79SYGBgnbO3adNGW7ZsUZcuXTR+/HhFRERo2rRpKi8vV7t27ST9FJMPPvigpk6dqqioKPn6+uo///M/L/jfBcCF2Sw+xAIAABiOI0QAAMB4BBEAADAeQQQAAIxHEAEAAOMRRAAAwHgEEQAAMB5BBAAAjEcQAQAA4xFEAADAeAQRAAAwHkEEAACM9/8A3kPAnAeJepkAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# to know how many students in hours\n",
    "import seaborn as sns\n",
    "x=df['Hours Studied']\n",
    "sns.histplot(x,color='red')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "3345c79c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Hours Studied', ylabel='Count'>"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkQAAAGwCAYAAABIC3rIAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABC40lEQVR4nO3de1yUZf7/8dfIGQQUFJBExSRFsZO65qHUPGV5WvtlpZlmx6+mkpplbhu5qZu7mZtulmVqkVm7q2VteU7NtdIo85B5SPMUSAcCVASF+/fHhWOkeIBhZuB+Px+PedTcc83M55aZ+37f13Xd9zgsy7IQERERsbFqni5ARERExNMUiERERMT2FIhERETE9hSIRERExPYUiERERMT2FIhERETE9hSIRERExPZ8PV1AZVFUVMQPP/xAaGgoDofD0+WIiIjIRbAsi9zcXGJjY6lWrfR+IAWii/TDDz8QFxfn6TJERESkDA4ePEjdunVLfVyB6CKFhoYC5h80LCzMw9WIiIjIxcjJySEuLs65Hy+NAtFFOj1MFhYWpkAkIiJSyVxouosmVYuIiIjtKRCJiIiI7SkQiYiIiO0pEImIiIjtKRCJiIiI7SkQiYiIiO0pEImIiIjtKRCJiIiI7SkQiYiIiO0pEImIiIjtKRCJiIiI7SkQiYiIiO0pEImIiIjtKRCJiIiI7fl6ugCBAwcO8NNPP7n0NWvVqkW9evVc+poiIiJVlQKRhx04cIDExCYcP57n0tcNDg5ix45vFYpEREQuggKRh/30008cP55HauofSUys7ZLX3LHjR+66azE//fSTApGIiMhFUCDyEomJtbn22jqeLkNEvJirh9c1tC5yhgKRiEglUBHD65VhaF1zLMVdFIhERCoBVw+vV4ahdc2xFHdSIBIRqUTsNLyuOZbiTgpEIuehORsinmenECieo0AkUgq7ztkQEbEjBSKRUthxzoaIiF0pEIlcgLrrRUTOVtWmFCgQiYiIyCWpilMKFIhERETkklTFKQUKRCIiIlImVWlKQTVPFyAiIiLiaQpEIiIiYnsKRCIiImJ7mkMkF0U/sCgiIlWZApFckH5gUUTEvaraNX4qAwUiuSD9wKK9qDdQxLOq4jV+KgMFIrloVen0Sjk3O/cG6ohcvEVVvMZPZaBAJCJOdu0N1BG5eCMdhLqXApGInMVuG2IdkYuIApGISDG7BUEROUPXIRIRERHbUyASERER21MgEhEREdtTIBIRERHbUyASERER21MgEhEREdtTIBIRERHbUyASERER21MgEhEREdtTIBIRERHbUyASERER21MgEhEREdtTIBIRERHbUyASERER2/NoIFq3bh29evUiNjYWh8PBu+++W+Jxy7JISUkhNjaWoKAgOnbsyPbt20u0yc/PZ8SIEdSqVYuQkBB69+7NoUOHSrTJyspi0KBBhIeHEx4ezqBBg/j1118reO1ERESksvBoIDp27BhXXXUVM2fOPOfjU6dOZdq0acycOZNNmzYRExND165dyc3NdbZJTk5m8eLFLFy4kPXr13P06FF69uxJYWGhs82AAQPYvHkzS5cuZenSpWzevJlBgwZV+PqJiIhI5eDryTfv0aMHPXr0OOdjlmUxffp0JkyYQL9+/QCYP38+0dHRLFiwgAcffJDs7GzmzJnDG2+8QZcuXQBITU0lLi6OlStX0r17d3bs2MHSpUv57LPPaN26NQCvvPIKbdq0YefOnTRu3Ng9KysiIiJey2vnEO3bt4+MjAy6devmXBYQEECHDh3YsGEDAGlpaZw8ebJEm9jYWJKSkpxtPv30U8LDw51hCOC6664jPDzc2eZc8vPzycnJKXETERGRqslrA1FGRgYA0dHRJZZHR0c7H8vIyMDf35+aNWuet01UVNRZrx8VFeVscy5TpkxxzjkKDw8nLi6uXOsjIiIi3strA9FpDoejxH3Lss5a9nu/b3Ou9hd6nfHjx5Odne28HTx48BIrFxERkcrCawNRTEwMwFm9OJmZmc5eo5iYGAoKCsjKyjpvmyNHjpz1+j/++ONZvU+/FRAQQFhYWImbiIiIVE1eG4ji4+OJiYlhxYoVzmUFBQWsXbuWtm3bAtCiRQv8/PxKtElPT2fbtm3ONm3atCE7O5uNGzc623z++edkZ2c724iIiIi9efQss6NHj7Jnzx7n/X379rF582YiIiKoV68eycnJTJ48mYSEBBISEpg8eTLBwcEMGDAAgPDwcO69917GjBlDZGQkERERjB07lubNmzvPOktMTOSmm27i/vvv5+WXXwbggQceoGfPnjrDTERERAAPB6IvvviCTp06Oe+PHj0agMGDBzNv3jzGjRtHXl4ew4YNIysri9atW7N8+XJCQ0Odz3n++efx9fWlf//+5OXl0blz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Tp06OBwOV9clIiIi4jZlCkQvvfQS8+bNY9CgQa6uR0RERMTtynQdooKCAtq2bevqWkREREQ8okyB6L777mPBggWurkVERETEI8o0ZHbixAlmz57NypUrufLKK/Hz8yvx+LRp01xSnIiIiIg7lCkQbdmyhauvvhqAbdu2lXhME6xFRESksinTkNnHH39c6m316tUuLfDw4cPcddddREZGEhwczNVXX01aWprzccuySElJITY2lqCgIDp27Mj27dtLvEZ+fj4jRoygVq1ahISE0Lt3bw4dOuTSOkVERKTyKlMgcpesrCzatWuHn58fH330Ed988w3PPfccNWrUcLaZOnUq06ZNY+bMmWzatImYmBi6du1Kbm6us01ycjKLFy9m4cKFrF+/nqNHj9KzZ89z/kCtiIiI2E+Zhsw6dep03qExV/USPfvss8TFxTF37lznst/+eKxlWUyfPp0JEybQr18/AObPn090dDQLFizgwQcfJDs7mzlz5vDGG2/QpUsXAFJTU4mLi2PlypV07979nO+dn59Pfn6+835OTo5L1klERES8T5l6iK6++mquuuoq561p06YUFBTw5Zdf0rx5c5cVt2TJElq2bMltt91GVFQU11xzDa+88orz8X379pGRkUG3bt2cywICAujQoQMbNmwAIC0tjZMnT5ZoExsbS1JSkrPNuUyZMoXw8HDnLS4uzmXrJSIiIt6lTD1Ezz///DmXp6SkcPTo0XIV9Ft79+5l1qxZjB49mieeeIKNGzcycuRIAgICuPvuu8nIyAAgOjq6xPOio6PZv38/ABkZGfj7+1OzZs2z2px+/rmMHz+e0aNHO+/n5OQoFImIiFRRZf4ts3O56667+MMf/sDf//53l7xeUVERLVu2ZPLkyQBcc801bN++nVmzZnH33Xc72/1++M6yrAue7XahNgEBAQQEBJSjehEREaksXDqp+tNPPyUwMNBlr1enTh2aNm1aYlliYiIHDhwAICYmBuCsnp7MzExnr1FMTAwFBQVkZWWV2kZERETsrUw9RKcnMJ9mWRbp6el88cUXPPnkky4pDKBdu3bs3LmzxLJdu3ZRv359AOLj44mJiWHFihVcc801gPlZkbVr1/Lss88C0KJFC/z8/FixYgX9+/cHID09nW3btjF16lSX1SoiIiKVV5kCUXh4eIn71apVo3HjxkycOLHE5OXy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karCq+cQiYiIiLiDApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J4CkYiIiNieApGIiIjYngKRiIiI2J6tAtGLL75IfHw8gYGBtGjRgk8++cTTJYmIiIgXsE0gevvtt0lOTmbChAl89dVXXH/99fTo0YMDBw54ujQRERHxMNsEomnTpnHvvfdy3333kZiYyPTp04mLi2PWrFmeLk1EREQ8zNfTBbhDQUEBaWlpPP744yWWd+vWjQ0bNpzzOfn5+eTn5zvvZ2dnA5CTk+PS2o4ePQpAWtoPHD1a4JLX3Lnzp+LXTHO+fvleb2fx63lvjQDVqlWjqKjIJa8Frl/vilhncO1662/tvX/rylAj6PPoCpXhb11RNR49etTl+9nTr2dZ1vkbWjZw+PBhC7D+97//lVg+adIk64orrjjnc5566ikL0E033XTTTTfdqsDt4MGD580KtughOs3hcJS4b1nWWctOGz9+PKNHj3beLyoq4pdffiEyMrLU55RFTk4OcXFxHDx4kLCwMJe9rjep6uuo9av8qvo6VvX1g6q/jlq/srMsi9zcXGJjY8/bzhaBqFatWvj4+JCRkVFieWZmJtHR0ed8TkBAAAEBASWW1ahRo6JKJCwsrEp+yH+rqq+j1q/yq+rrWNXXD6r+Omr9yiY8PPyCbWwxqdrf358WLVqwYsWKEstXrFhB27ZtPVSViIiIeAtb9BABjB49mkGDBtGyZUvatGnD7NmzOXDgAA899JCnSxMREREPs00guv322/n555+ZOHEi6enpJCUl8eGHH1K/fn2P1hUQEMBTTz111vBcVVLV11HrV/lV9XWs6usHVX8dtX4Vz2FZFzoPTURERKRqs8UcIhEREZHzUSASERER21MgEhEREdtTIBIRERHbUyDykHXr1tGrVy9iY2NxOBy8++67ni7JpaZMmUKrVq0IDQ0lKiqKvn37On/7pqqYNWsWV155pfNCYm3atOGjjz7ydFkVZsqUKTgcDpKTkz1dikukpKTgcDhK3GJiYjxdlssdPnyYu+66i8jISIKDg7n66qtJS0vzdFku0aBBg7P+hg6Hg+HDh3u6NJc4deoUf/rTn4iPjycoKIiGDRsyceJEl/5umjfIzc0lOTmZ+vXrExQURNu2bdm0aZPb67DNaffe5tixY1x11VXcc8893HrrrZ4ux+XWrl3L8OHDadWqFadOnWLChAl069aNb775hpCQEE+X5xJ169blr3/9K40aNQJg/vz59OnTh6+++opmzZp5uDrX2rRpE7Nnz+bKK6/0dCku1axZM1auXOm87+Pj48FqXC8rK4t27drRqVMnPvroI6Kiovjuu+8q9Kr77rRp0yYKCwud97dt20bXrl257bbbPFiV6zz77LO89NJLzJ8/n2bNmvHFF19wzz33EB4ezqhRozxdnsvcd999bNu2jTfeeIPY2FhSU1Pp0qUL33zzDZdddpn7CnHJr6dKuQDW4sWLPV1GhcrMzLQAa+3atZ4upULVrFnTevXVVz1dhkvl5uZaCQkJ1ooVK6wOHTpYo0aN8nRJLvHUU09ZV111lafLqFCPPfaY1b59e0+X4TajRo2yLr/8cquoqMjTpbjELbfcYg0dOrTEsn79+ll33XWXhypyvePHj1s+Pj7WBx98UGL5VVddZU2YMMGttWjITNwiOzsbgIiICA9XUjEKCwtZuHAhx44do02bNp4ux6WGDx/OLbfcQpcuXTxdisvt3r2b2NhY4uPjueOOO9i7d6+nS3KpJUuW0LJlS2677TaioqK45ppreOWVVzxdVoUoKCggNTWVoUOHuvQHuD2pffv2rFq1il27dgHw9ddfs379em6++WYPV+Y6p06dorCwkMDAwBLLg4KCWL9+vVtr0ZCZVDjLshg9ejTt27cnKSnJ0+W41NatW2nTpg0nTpygevXqLF68mKZNm3q6LJdZuHAhX375pUfG8yta69atef3117niiis4cuQIzzzzDG3btmX79u1ERkZ6ujyX2Lt3L7NmzWL06NE88cQTbNy4kZEjRxIQEMDdd9/t6fJc6t133+XXX39lyJAhni7FZR577DGys7Np0qQJPj4+FBYWMmnSJO68805Pl+YyoaGhtGnThr/85S8kJiYSHR3NW2+9xeeff05CQoJ7i3Frf5ScE1V8yGzYsGFW/fr1rYMHD3q6FJfLz8+3du/ebW3atMl6/PHHrVq1alnbt2/3dFkuceDAASsqKsravHmzc1lVGjL7vaNHj1rR0dHWc8895+lSXMbPz89q06ZNiWUjRoywrrvuOg9VVHG6detm9ezZ09NluNRbb71l1a1b13rrrbesLVu2WK+//roVERFhzZs3z9OludSePXusG264wQIsHx8fq1WrVtbAgQOtxMREt9ahHiKpUCNGjGDJkiWsW7eOunXrerocl/P393dOqm7ZsiWbNm3iH//4By+//LKHKyu/tLQ0MjMzadGihXNZYWEh69atY+bMmeTn51epScghISE0b96c3bt3e7oUl6lTp85ZPZaJiYn85z//8VBFFWP//v2sXLmSRYsWeboUl3r00Ud5/PHHueOOOwBo3rw5+/fvZ8qUKQwePNjD1bnO5Zdfztq1azl27Bg5OTnUqVOH22+/nfj4eLfWoUAkFcKyLEaMGMHixYtZs2aN2z/YnmJZFvn5+Z4uwyU6d+7M1q1bSyy75557aNKkCY899liVCkMA+fn57Nixg+uvv97TpbhMu3btzrrcxa5duzz+o9auNnfuXKKiorjllls8XYpLHT9+nGrVSk719fHxqXKn3Z8WEhJCSEgIWVlZLFu2jKlTp7r1/RWIPOTo0aPs2bPHeX/fvn1s3ryZiIgI6tWr58HKXGP48OEsWLCA9957j9DQUDIyMgAIDw8nKCjIw9W5xhNPPEGPHj2Ii4sjNzeXhQsXsmbNGpYuXerp0lwiNDT0rDlfISEhREZGVom5YGPHjqVXr17Uq1ePzMxMnnnmGXJycqrUkfcjjzxC27ZtmTx5Mv3792fjxo3Mnj2b2bNne7o0lykqKmLu3LkMHjwYX9+qtUvr1asXkyZNol69ejRr1oyvvvqKadOmMXToUE+X5lLLli3DsiwaN27Mnj17ePTRR2ncuDH33HOPewtx6wCdOH388ccWcNZt8ODBni7NJc61boA1d+5cT5fmMkOHDrXq169v+fv7W7Vr17Y6d+5sLV++3NNlVaiqNIfo9ttvt+rUqWP5+flZsbGxVr9+/arM/K/fev/9962kpCQrICDAatKkiTV79mxPl+RSy5YtswBr586dni7F5XJycqxRo0ZZ9erVswIDA62GDRtaEyZMsPLz8z1dmku9/fbbVsOGDS1/f38rJibGGj58uPXrr7+6vQ6HZVmWeyOYiIiIiHfRdYhERETE9hSIRERExPYUiERERMT2FIhERETE9hSIRERExPYUiERERMT2FIhERETE9hSIRERExPYUiERE3KRBgwZMnz7ded/hcPDuu++W6zWHDBlC3759y/UaIqJAJCIuUtqOec2aNTgcDn799Ve313Qhe/fu5c477yQ2NpbAwEDq1q1Lnz592LVrFwDff/89DoeDzZs3V8j7p6en06NHjwp5bRG5NFXrl/BExLZOnjyJn5/fRbcvKCiga9euNGnShEWLFlGnTh0OHTrEhx9+SHZ2dgVWekZMTIxb3kdELkw9RCLidv/5z39o1qwZAQEBNGjQgOeee67E4+caSqpRowbz5s0DzvTcvPPOO3Ts2JHAwEBSU1PZv38/vXr1ombNmoSEhNCsWTM+/PDDc9bwzTffsHfvXl588UWuu+466tevT7t27Zg0aRKtWrUCID4+HoBrrrkGh8NBx44dAejYsSPJycklXq9v374MGTLEeT8zM5NevXoRFBREfHw8b7755lk1/H49Dx8+zO23307NmjWJjIykT58+fP/9987HCwsLGT16NDVq1CAyMpJx48ahn6MUcQ0FIhFxq7S0NPr3788dd9zB1q1bSUlJ4cknn3SGnUvx2GOPMXLkSHbs2EH37t0ZPnw4+fn5rFu3jq1bt/Lss89SvXr1cz63du3aVKtWjX//+98UFhaes83GjRsBWLlyJenp6SxatOiiaxsyZAjff/89q1ev5t///jcvvvgimZmZpbY/fvw4nTp1onr16qxbt47169dTvXp1brrpJgoKCgB47rnneO2115gzZw7r16/nl19+YfHixRddk4iUTkNmIuIyH3zwwVkB5PdhY9q0aXTu3Jknn3wSgCuuuIJvvvmGv/3tbyV6WC5GcnIy/fr1c94/cOAAt956K82bNwegYcOGpT73sssu44UXXmDcuHE8/fTTtGzZkk6dOjFw4EDn82rXrg1AZGTkJQ1v7dq1i48++ojPPvuM1q1bAzBnzhwSExNLfc7ChQupVq0ar776Kg6HA4C5c+dSo0YN1qxZQ7du3Zg+fTrjx4/n1ltvBeCll15i2bJlF12XiJROPUQi4jKdOnVi8+bNJW6vvvpqiTY7duygXbt2JZa1a9eO3bt3l9pTU5qWLVuWuD9y5EieeeYZ2rVrx1NPPcWWLVvO+/zhw4eTkZFBamoqbdq04V//+hfNmjVjxYoVl1TH7+3YsQNfX98S9TVp0oQaNWqU+py0tDT27NlDaGgo1atXp3r16kRERHDixAm+++47srOzSU9Pp02bNs7n/P49RKTsFIhExGVCQkJo1KhRidtll11Woo1lWc4ekN8u+y2Hw3HWspMnT57z/X7rvvvuY+/evQwaNIitW7fSsmVLZsyYcd6aQ0ND6d27N5MmTeLrr7/m+uuv55lnnjnvc6pVq3be+k4/9vv1PJ+ioiJatGhxVqDctWsXAwYMuOjXEZGyUSASEbdq2rQp69evL7Fsw4YNXHHFFfj4+ABmqCo9Pd35+O7duzl+/PhFvX5cXBwPPfQQixYtYsyYMbzyyisXXZvD4aBJkyYcO3YMAH9/f+DsYb/f11dYWMi2bduc9xMTEzl16hRffPGFc9nOnTvPe+mBa6+9lt27dxMVFXVWqAwPDyc8PJw6derw2WefOZ9z6tQp0tLSLnr9RKR0CkQi4lZjxoxh1apV/OUvf2HXrl3Mnz+fmTNnMnbsWGebG2+8kZkzZ/Lll1/yxRdf8NBDD13UKfXJycksW7aMffv28eWXX7J69epS5+1s3ryZPn368O9//5tvvvmGPXv2MGfOHF577TX69OkDQFRUFEFBQSxdupQjR444T8e/8cYb+e9//8t///tfvv32W4YNG1Yi7DRu3JibbrqJ+++/n88//5y0tDTuu+8+goKCSq194MCB1KpViz59+vDJJ5+wb98+1q5dy6hRozh06BAAo0aN4q9//SuLFy8+5/uKSNkpEImIW1177bW88847LFy4kKSkJP785z8zceLEEhOqn3vuOeLi4rjhhhsYMGAAY8eOJTg4+IKvXVhYyPDhw0lMTOSmm26icePGvPjii+dsW7duXRo0aMDTTz9N69atufbaa/nHP/7B008/zYQJEwAzR+eFF17g5ZdfJjY21hmUhg4dyuDBg7n77rvp0KED8fHxdOrUqcTrz507l7i4ODp06EC/fv144IEHiIqKKrX24OBg1q1bR7169ejXrx+JiYkMHTqUvLw8wsLCABMm7777boYMGUKbNm0IDQ3lj3/84wX/XUTkwhyWLmIhIiIiNqceIhEREbE9BSIRERGxPQUiERERsT0FIhEREbE9BSIRERGxPQUiERERsT0FIhEREbE9BSIRERGxPQUiERERsT0FIhEREbE9BSIRERGxvf8PBP3Z2YVsUzQAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(x,color='yellow',kde=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "id": "056b3a34",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Hours Studied', ylabel='Performance Index'>"
      ]
     },
     "execution_count": 155,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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y8tTdHS0Dh06pFq1avn03vz8/HLNafEFc1oAACjN29/fzNkBAAC2xjBWBTGnBQCA4EbYqSDmtAAAENwYxgIAALZG2AEAALZG2AEAALZG2AEAALZG2AEAALZG2AEAALZG2AEAALZG2AEAALZG2AEAALZG2AEAALZG2AEAALZG2AEAALZG2AEAALbGrueSjDGSpLy8vABXAgAAvHX89/bx3+NlIexIOnz4sCQpMTExwJUAAABfHT58WNHR0WW+7jCni0NVQFFRkfbv36+oqCg5HA6/flZeXp4SExP1448/qlatWn79rECw8/VxbZUT11Y5cW2V05m+NmOMDh8+rISEBFWrVvbMHHp2JFWrVk3169c/o59Zq1Yt2/1PXpydr49rq5y4tsqJa6uczuS1napH5zgmKAMAAFsj7AAAAFsj7Jxh4eHheuihhxQeHh7oUvzCztfHtVVOXFvlxLVVTsF6bUxQBgAAtkbPDgAAsDXCDgAAsDXCDgAAsDXCDgAAsDXCzhm0Zs0a9erVSwkJCXI4HHrjjTcCXZIlpk2bprZt2yoqKkqxsbHq06ePtm7dGuiyLDF79my1bt3as0BW+/bt9d577wW6LL+YNm2aHA6HUlNTA12KJdLT0+VwOEo84uPjA12WZfbt26dbb71VMTExql69ui688EJt3Lgx0GVVWMOGDUv9d3M4HBo5cmSgS6uwgoICTZ48WcnJyYqMjFSjRo00ZcoUFRUVBbo0Sxw+fFipqalq0KCBIiMjlZKSovXr1we6LEmsoHxG/fnnn7rgggs0ePBgXX/99YEuxzKrV6/WyJEj1bZtWxUUFGjSpEnq3r27vv32W9WoUSPQ5VVI/fr19eijj6px48aSpPnz56t379764osv1KJFiwBXZ53169drzpw5at26daBLsVSLFi30wQcfeJ6HhIQEsBrrHDx4UB06dFDnzp313nvvKTY2Vj/88INq164d6NIqbP369SosLPQ8/+abb9StWzfdcMMNAazKGtOnT9fTTz+t+fPnq0WLFtqwYYMGDx6s6OhojR07NtDlVdgdd9yhb775Ri+++KISEhL00ksvqWvXrvr2229Vr169wBZnEBCSzLJlywJdhl/k5uYaSWb16tWBLsUvzjrrLPPcc88FugzLHD582DRp0sSsXLnSdOzY0YwdOzbQJVnioYceMhdccEGgy/CLCRMm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33yzvv76a6Wnp+uBBx7wBBlfTJgwQWPGjNF3332nHj16aOTIkcrPz9eaNWv09ddfa/r06apZs+ZJ33vOOeeoWrVqWrx4sQoLC096zOeffy5J+uCDD3TgwAEtXbrU69oGDRqk3bt366OPPtLixYs1a9Ys5ebmlnn8kSNH1LlzZ9WsWVNr1qzR2rVrVbNmTV111VVyuVySpMcff1wvvPCCnn/+ea1du1a//fabli1b5nVNAMrGMBYAr7z99tulwsWJQWLGjBnq0qWLHnjgAUnSeeedp2+//Vb/8z//U6JnxBupqanq27ev5/nevXt1/fXXq1WrVpKkRo0alfneevXq6cknn1RaWpoefvhhtWnTRp07d1b//v097zvnnHMkSTExMT4NOW3btk3vvfeePv30U7Vr106S9Pzzz6tZs2ZlvmfRokWqVq2annvuOTkcDknS3LlzVbt2ba1atUrdu3dXdna2Jk6cqOuvv16S9PTTT2v58uVe1wWgbPTsAPBK586dtXnz5hKP5557rsQx3333nTp06FCirUOHDtq+fXuZPSxladOmTYnnY8aMUUZGhjp06KCHHnpIX3311SnfP3LkSOXk5Oill15S+/bt9frrr6tFixZauXKlT3Wc6LvvvlNoaGiJ+po2baratWuX+Z6NGzdqx44dioqKUs2aNVWzZk3VqVNHf/31l3744QcdOnRIBw4cUPv27T3vOfEzAJQfYQeAV2rUqKHGjRuXeNSrV6/EMcYYT89F8bbiHA5HqTa3233Szyvujjvu0M6dO3Xbbbfp66+/Vps2bfTUU0+dsuaoqChdd911mjp1qr788ktdfvnlysjIOOV7qlWrdsr6jr924nWeSlFRkS655JJSYXHbtm3q16+f1+cBUD6EHQCWad68udauXVuibd26dTrvvPMUEhIi6djw0YEDBzyvb9++XUeOHPHq/ImJiRo2bJiWLl2qe+65R88++6zXtTkcDjVt2lR//vmnJMnpdEoqPRR3Yn2FhYX65ptvPM+bNWumgoICbdiwwdO2devWU95+f/HFF2v79u2KjY0tFRijo6MVHR2tunXr6tNPP/W8p6CgQBs3bvT6+gCUjbADwDL33HOPPvzwQz3yyCPatm2b5s+fr5kzZ2r8+PGeY6688krNnDlTmzZt0oYNGzRs2DCvbitPTU3V8uXLtWvXLm3atEkfffRRmfNkNm/erN69e2vx4sX69ttvtWPHDj3//PN64YUX1Lt3b0lSbGysIiMj9f777+vnn3/23JJ+5ZVX6p133tE777yj77//XiNGjCgRZM4//3xdddVVuvPOO/XZZ59p48aNuuOOOxQZGVlm7f3799fZZ5+t3r1765NPPtGuXbu0evVqjR07Vj/99JMkaezYsXr00Ue1bNmyk34ugPIj7ACwzMUXX6zXXntNixYtUsuWLfXggw9qypQpJSYnP/7440pMTNQVV1yhfv36afz48apevfppz11YWKiRI0eqWbNmuuqqq3T++edr1qxZJz22fv36atiwoR5++GG1a9dOF198sZ544gk9/PDDmjRpkqRjc2KefPJJPfPMM0pISPCEoCFDhmjgwIEaMGCAOnbsqOTkZHXu3LnE+efOnavExER17NhRffv21V133aXY2Ngya69evbrWrFmjpKQk9e3bV82aNdOQIUN09OhR1apVS9KxoDhgwAANGjRI7du3V1RUlP75z3+e9s8FwOk5DAs5AAAAG6NnBwAA2BphBwAA2BphBwAA2BphBwAA2BphBwAA2BphBwAA2BphBwAA2BphBwAA2BphBwAA2BphBwAA2BphBwAA2Nr/Ay45c8ntEtffAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(x=df['Hours Studied'],y=df['Performance Index'],color='violet')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "id": "92bd56ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='Extracurricular Activities', ylabel='count'>"
      ]
     },
     "execution_count": 156,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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/fRTn/FDhw7pzjvv9Bm788479cUXX6ioqKgiW0QlQCCC0QICAhQQ8OOv9Lt48aLuu+8+eTwen+WLL75Q165dVb16dW3atEn/+Mc/1Lp1a73yyitq2bKlMjMz/XwWAK6ma9eu6tOnj5599lmfccuyij1xxkUTcxnzy12Bn3P77bdr3bp1atq0qR2SLudwOHTnnXfqzjvv1HPPPacmTZooNTVV48aNq+BuAVyPF198Ue3bt9ctt9xij7Vu3Vrbtm3zqduxY4duueUWVa9evaJbhJ8xQwT8nyeeeELfffedHnnkEe3evVtff/21Nm7cqOHDh6uoqEgZGRlKSkrShx9+qKNHj+qtt95STk6OWrVq5e/WAfyMmJgYDRkyRK+88oo9Nn78eH3wwQd64YUX9Pnnn2v58uVKSUnRhAkT/Ngp/IVABPyfiIgIbd++XUVFRerTp4+io6P11FNPyeVyqVq1agoJCdE///lP3Xvvvbrlllv0pz/9SbNnz1a/fv383TqAa/DCCy/4XBK7/fbb9eabb2rNmjWKjo7Wc889p+eff17Dhg3zX5PwG54yAwAAxmOGCAAAGI9ABAAAjEcgAgAAxiMQAQAA4xGIAACA8QhEAADAeAQiAABgPAIRAAAwHoEIQJWzZcsWORwOnT59usz26XA4tH79+jLbX0W43p6bNm2qefPmXbVm2rRpat++/S/qC6iKCERAFTRs2DA5HI5iS9++fa95H927d1dCQkL5NVmOunTpoqysLLlcLn+3cl1atmypwMBA/b//9/+u63NXCilZWVnX9atj9uzZo1GjRtnrJQWqCRMm6IMPPriu/oAbAYEIqKL69u2rrKwsn+X1118v02NYlqULFy6U6T6v1fnz5684HhgYKLfbLYfDUcFdXdmV+r1k27Zt+uGHH/Twww9r2bJlZXJMt9stp9N5zfWhoaGqVavWVWtq166tBg0a/NLWgCqHQARUUU6nU26322epV6+epB8vKQUGBmrr1q12/ezZs9WwYUNlZWVp2LBhSk9P1/z58+3ZpSNHjtiXot577z117NhRTqdTW7du1VdffaX7779f4eHhql27tu644w69//77Pv0UFBQoMTFRkZGRcjqdioqK0pIlSyRJy5YtU926dX3q169f7xNoLs2CvPbaa2revLmcTqcsy5LD4dBf/vIX3X///QoODtaMGTNKvGS2fft2devWTbVq1VK9evXUp08f5ebmSir5UlH79u01bdq0K36/EydO1C233KJatWqpefPmmjJlik/ouVK/V7JkyRLFxcUpPj5er732WrHa48ePa/Dgwapfv76Cg4PVsWNHZWRkaNmyZZo+fbo+/vhj++/qUqD66QxPbGys/vjHP/rsMycnRzVq1NDmzZuLfQ9NmzaVJD344INyOBz2ekmzUUuXLlWrVq1Us2ZN3XrrrXr11VftbYWFhXryySfVqFEj1axZU02bNlVycvIVvwegsgrwdwMAyt6ly2Hx8fH6+OOPdeTIEU2ePFmvv/66GjVqpPnz5+vzzz9XdHS0nn/+eUk/zh4cOXJEkpSYmKj//M//VPPmzVW3bl0dP35c9957r2bMmKGaNWtq+fLluu+++3T48GE1btxYkvToo49q586devnll9WuXTtlZmbq1KlT19X3l19+qTfffFPr1q1T9erV7fGpU6cqOTlZc+fOVfXq1ZWZmenzOY/Hox49emj48OF6+eWXFRAQoM2bN6uoqKjU32GdOnW0bNkyRUREaP/+/Ro5cqTq1KmjxMTEn+33cmfOnNHf/vY3ZWRk6NZbb9W5c+e0ZcsW3X333ZKks2fPqlu3brrpppu0YcMGud1uffTRR7p48aIGDRqkAwcOKC0tzQ6hJV0qHDJkiF566SUlJyfbQfONN95QeHi4unXrVqx+z549CgsL09KlS9W3b98r9r948WJNnTpVKSkpuu2227Rv3z6NHDlSwcHBGjp0qF5++WVt2LBBb775pho3bqxjx47p2LFj1/5FA5UEgQioot5++23Vrl3bZ2zixImaMmWKJGnGjBl6//33NWrUKB08eFDx8fF68MEHJf34AzUwMFC1atWS2+0utu/nn39evXr1stcbNGigdu3a2eszZsxQamqqNmzYoCeffFKff/653nzzTW3atEk9e/aUJDVv3vy6z6mwsFArVqxQaGioz3hcXJyGDx9ur18eiGbNmqWOHTv6zFy0adPmuo//U3/605/sPzdt2lTjx4/XG2+84ROIrtTv5dasWaOoqCi7p8GDB2vJkiV2IFq9erVycnK0Z88e1a9fX5LUokUL+/O1a9dWQEBAiX9XlwwaNEhPP/20tm3bprvuusveb1xcnKpVK34x4FLPdevWvep+X3jhBc2ePVsPPfSQJKlZs2b69NNPtXDh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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(x=df['Extracurricular Activities'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "id": "973fdd14",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Hours Studied', 'Previous Scores', 'Extracurricular Activities',\n",
       "       'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 157,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "id": "6bddbfe7",
   "metadata": {},
   "outputs": [],
   "source": [
    "x=df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities',\n",
    "       'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index']]\n",
    "      "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "id": "535f7952",
   "metadata": {},
   "outputs": [],
   "source": [
    "y=df['Performance Index']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "id": "052a75b6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000, 6)"
      ]
     },
     "execution_count": 166,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "id": "38b58a26",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours Studied</th>\n",
       "      <th>Previous Scores</th>\n",
       "      <th>Extracurricular Activities</th>\n",
       "      <th>Sleep Hours</th>\n",
       "      <th>Sample Question Papers Practiced</th>\n",
       "      <th>Performance Index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7</td>\n",
       "      <td>99</td>\n",
       "      <td>Yes</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>91.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4</td>\n",
       "      <td>82</td>\n",
       "      <td>No</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>65.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>51</td>\n",
       "      <td>Yes</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>45.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5</td>\n",
       "      <td>52</td>\n",
       "      <td>Yes</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>36.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7</td>\n",
       "      <td>75</td>\n",
       "      <td>No</td>\n",
       "      <td>8</td>\n",
       "      <td>5</td>\n",
       "      <td>66.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours Studied  Previous Scores Extracurricular Activities  Sleep Hours  \\\n",
       "0              7               99                        Yes            9   \n",
       "1              4               82                         No            4   \n",
       "2              8               51                        Yes            7   \n",
       "3              5               52                        Yes            5   \n",
       "4              7               75                         No            8   \n",
       "\n",
       "   Sample Question Papers Practiced  Performance Index  \n",
       "0                                 1               91.0  \n",
       "1                                 2               65.0  \n",
       "2                                 2               45.0  \n",
       "3                                 2               36.0  \n",
       "4                                 5               66.0  "
      ]
     },
     "execution_count": 160,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "id": "951eed2d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(10000,)"
      ]
     },
     "execution_count": 167,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "id": "80012f0a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Hours Studied</th>\n",
       "      <th>Previous Scores</th>\n",
       "      <th>Extracurricular Activities</th>\n",
       "      <th>Sleep Hours</th>\n",
       "      <th>Sample Question Papers Practiced</th>\n",
       "      <th>Performance Index</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>7</td>\n",
       "      <td>99</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>91.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4</td>\n",
       "      <td>82</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>65.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>8</td>\n",
       "      <td>51</td>\n",
       "      <td>1</td>\n",
       "      <td>7</td>\n",
       "      <td>2</td>\n",
       "      <td>45.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>5</td>\n",
       "      <td>52</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>2</td>\n",
       "      <td>36.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>7</td>\n",
       "      <td>75</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>5</td>\n",
       "      <td>66.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Hours Studied  Previous Scores  Extracurricular Activities  Sleep Hours  \\\n",
       "0              7               99                           1            9   \n",
       "1              4               82                           0            4   \n",
       "2              8               51                           1            7   \n",
       "3              5               52                           1            5   \n",
       "4              7               75                           0            8   \n",
       "\n",
       "   Sample Question Papers Practiced  Performance Index  \n",
       "0                                 1               91.0  \n",
       "1                                 2               65.0  \n",
       "2                                 2               45.0  \n",
       "3                                 2               36.0  \n",
       "4                                 5               66.0  "
      ]
     },
     "execution_count": 164,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Extracurricular Activities']=df['Extracurricular Activities'].apply(lambda x:1 if x==\"Yes\"else 0)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "id": "50756038",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-9 {color: black;}#sk-container-id-9 pre{padding: 0;}#sk-container-id-9 div.sk-toggleable {background-color: white;}#sk-container-id-9 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-9 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-9 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-9 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-9 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-9 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-9 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-9 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-9 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-9 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-9 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-9 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-9 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-9 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-9 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-9 div.sk-item {position: relative;z-index: 1;}#sk-container-id-9 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-9 div.sk-item::before, #sk-container-id-9 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-9 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-9 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-9 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-9 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-9 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-9 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-9 div.sk-label-container {text-align: center;}#sk-container-id-9 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-9 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-9\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-9\" type=\"checkbox\" checked><label for=\"sk-estimator-id-9\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 172,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_splitfrom sklearn.model_selection import train_test_split\n",
    "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)\n",
    "from sklearn.linear_model import LinearRegression\n",
    "model=LinearRegression()\n",
    "model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 224,
   "id": "74affc62",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-14 {color: black;}#sk-container-id-14 pre{padding: 0;}#sk-container-id-14 div.sk-toggleable {background-color: white;}#sk-container-id-14 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-14 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-14 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-14 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-14 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-14 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-14 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-14 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-14 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-14 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-14 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-14 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-14 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-14 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-14 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-14 div.sk-item {position: relative;z-index: 1;}#sk-container-id-14 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-14 div.sk-item::before, #sk-container-id-14 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-14 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-14 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-14 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-14 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-14 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-14 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-14 div.sk-label-container {text-align: center;}#sk-container-id-14 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-14 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-14\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-14\" type=\"checkbox\" checked><label for=\"sk-estimator-id-14\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">LinearRegression</label><div class=\"sk-toggleable__content\"><pre>LinearRegression()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "LinearRegression()"
      ]
     },
     "execution_count": 224,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(x_train,y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 225,
   "id": "9a1b5017",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.18257493, -0.01851946,  0.24274054,  1.51014887,  1.18647929,\n",
       "        1.60639492,  1.38857942,  1.20936669,  1.71384936, -0.01382608,\n",
       "        1.75028062, -0.20437734,  0.06495741,  1.92069806,  1.59072017,\n",
       "        1.12769583,  1.1459185 ,  1.16466959, -0.01255212,  1.46076414,\n",
       "        1.02528253, -0.09999924,  1.17189317,  1.17132566,  1.14284581,\n",
       "        1.07593758,  0.9223231 ,  2.09121803,  0.01070126,  0.14320053])"
      ]
     },
     "execution_count": 225,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=model.predict(x_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 227,
   "id": "3fa9fb6b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9666666666666667"
      ]
     },
     "execution_count": 227,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import accuracy_score\n",
    "acc=accuracy_score(y_test,np.round(y_pred))\n",
    "acc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 235,
   "id": "f5d7a596",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-4.62878021])"
      ]
     },
     "execution_count": 235,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x=df[['Hours Studied', 'Previous Scores', 'Extracurricular Activities',\n",
    "       'Sleep Hours', 'Sample Question Papers Practiced', ]]\n",
    "data=[[8,85,1,6]]\n",
    "prediction=model.predict(data)\n",
    "prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 236,
   "id": "7cfa819a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-15 {color: black;}#sk-container-id-15 pre{padding: 0;}#sk-container-id-15 div.sk-toggleable {background-color: white;}#sk-container-id-15 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-15 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-15 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-15 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-15 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-15 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-15 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-15 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-15 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-15 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-15 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-15 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-15 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-15 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-15 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-15 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-15 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-15 div.sk-item {position: relative;z-index: 1;}#sk-container-id-15 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-15 div.sk-item::before, #sk-container-id-15 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-15 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-15 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-15 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-15 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-15 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-15 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-15 div.sk-label-container {text-align: center;}#sk-container-id-15 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-15 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-15\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>Ridge()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-15\" type=\"checkbox\" checked><label for=\"sk-estimator-id-15\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">Ridge</label><div class=\"sk-toggleable__content\"><pre>Ridge()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "Ridge()"
      ]
     },
     "execution_count": 236,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import Ridge\n",
    "clf=Ridge()\n",
    "clf.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 238,
   "id": "e6ac45a7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-0.16457075, -0.03851152,  0.21074426,  1.56273831,  1.20558724,\n",
       "        1.58424073,  1.36638854,  1.19486691,  1.67965026, -0.02128002,\n",
       "        1.7272267 , -0.16626142,  0.04279544,  1.96529935,  1.58017503,\n",
       "        1.13521009,  1.15610724,  1.17658102, -0.0128643 ,  1.53096035,\n",
       "        1.0262604 , -0.08725997,  1.20276832,  1.17378262,  1.15335693,\n",
       "        1.0527963 ,  0.90985843,  2.01723857,  0.0139326 ,  0.10315806])"
      ]
     },
     "execution_count": 238,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred=clf.predict(x_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 239,
   "id": "c15b9785",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.918557504600991"
      ]
     },
     "execution_count": 239,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf.score(x_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f8c4a127",
   "metadata": {},
   "outputs": [],
   "source": [
    "#unsupervised machine learning\n",
    "it is also known as unsupervised"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
