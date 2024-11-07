{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "MTIXrkCYKtlu"
   },
   "source": [
    "Importing the Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "mL7HLYQFXW-c"
   },
   "outputs": [],
   "source": [
    "import numpy as np  # type: ignore\n",
    "import pandas as pd # type: ignore\n",
    "import matplotlib.pyplot as plt # type: ignore\n",
    "import seaborn as sns # type: ignore\n",
    "from sklearn.cluster import KMeans # type: ignore\n",
    "from sklearn.preprocessing import LabelEncoder # type: ignore"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "KigjC6mBLJN3"
   },
   "source": [
    "Data Collection & Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "id": "gTSFo2LiLIav"
   },
   "outputs": [],
   "source": [
    "# loading the data from csv file to a Pandas DataFrame\n",
    "customer_data = pd.read_csv('customers.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "customer_data.fillna(customer_data.mean(), inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Missing values after handling:\n",
      "Age                     0\n",
      "Purchase Frequency      0\n",
      "Average Amount Spent    0\n",
      "Spending Score          0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(\"Missing values after handling:\")\n",
    "print(customer_data.isnull().sum())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = customer_data[['Age', 'Average Amount Spent', 'Spending Score']].values "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Any NaN values in X: False\n"
     ]
    }
   ],
   "source": [
    "print(\"Any NaN values in X:\", np.isnan(X).any())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 196
    },
    "id": "mbtjztN3Lhcu",
    "outputId": "4b5e3ec9-1784-4918-bece-7616a2305e4b"
   },
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
       "      <th>Age</th>\n",
       "      <th>Purchase Frequency</th>\n",
       "      <th>Average Amount Spent</th>\n",
       "      <th>Spending Score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>231</th>\n",
       "      <td>58</td>\n",
       "      <td>6</td>\n",
       "      <td>15000.0</td>\n",
       "      <td>90000.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>232</th>\n",
       "      <td>24</td>\n",
       "      <td>6</td>\n",
       "      <td>24600.0</td>\n",
       "      <td>147600.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>233</th>\n",
       "      <td>89</td>\n",
       "      <td>2</td>\n",
       "      <td>20000.0</td>\n",
       "      <td>40000.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>234</th>\n",
       "      <td>55</td>\n",
       "      <td>5</td>\n",
       "      <td>55555.0</td>\n",
       "      <td>277775.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>235</th>\n",
       "      <td>55</td>\n",
       "      <td>5</td>\n",
       "      <td>55555.0</td>\n",
       "      <td>277775.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Age  Purchase Frequency  Average Amount Spent  Spending Score\n",
       "231   58                   6               15000.0         90000.0\n",
       "232   24                   6               24600.0        147600.0\n",
       "233   89                   2               20000.0         40000.0\n",
       "234   55                   5               55555.0        277775.0\n",
       "235   55                   5               55555.0        277775.0"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# first 5 rows in the dataframe\n",
    "customer_data.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "-NWZTDNRLofr",
    "outputId": "5cdb9e6b-20f2-4676-da6c-014dc7b3a38e"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(236, 4)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# finding the number of rows and columns\n",
    "customer_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "K5qKxwHiL56b",
    "outputId": "ce5ec885-f107-493a-ecb7-42170d1fd7d5"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 236 entries, 0 to 235\n",
      "Data columns (total 4 columns):\n",
      " #   Column                Non-Null Count  Dtype  \n",
      "---  ------                --------------  -----  \n",
      " 0   Age                   236 non-null    int64  \n",
      " 1   Purchase Frequency    236 non-null    int64  \n",
      " 2   Average Amount Spent  236 non-null    float64\n",
      " 3   Spending Score        236 non-null    float64\n",
      "dtypes: float64(2), int64(2)\n",
      "memory usage: 7.5 KB\n"
     ]
    }
   ],
   "source": [
    "# getting some informations about the dataset\n",
    "customer_data.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "vBtCZvyFMI1O",
    "outputId": "d8a19c04-d0d1-4fb6-ea71-53e2137e0307"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Age                     0\n",
       "Purchase Frequency      0\n",
       "Average Amount Spent    0\n",
       "Spending Score          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# checking for missing values\n",
    "customer_data.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mDtt8uP0MoiH"
   },
   "source": [
    "Choosing the Annual Income Column & Spending Score column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "5vpIqX5qNHEB",
    "outputId": "ff589c21-c7ca-4c37-da4a-59ccddfbb828"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[5.50000e+01 2.46370e+04 7.39110e+01]\n",
      " [6.50000e+01 7.68480e+04 2.30544e+02]\n",
      " [8.00000e+01 3.00000e+04 1.80000e+02]\n",
      " [4.80000e+01 4.34310e+04 1.30293e+02]\n",
      " [2.50000e+01 2.50000e+04 8.00000e+01]\n",
      " [3.20000e+01 1.50000e+04 5.00000e+01]\n",
      " [1.80000e+01 1.80000e+04 6.00000e+01]\n",
      " [4.00000e+01 5.00000e+04 9.50000e+01]\n",
      " [2.70000e+01 3.00000e+04 8.50000e+01]\n",
      " [2.20000e+01 2.00000e+04 7.00000e+01]\n",
      " [3.50000e+01 2.20000e+04 6.50000e+01]\n",
      " [3.00000e+01 1.50000e+04 4.50000e+01]\n",
      " [1.90000e+01 1.70000e+04 5.50000e+01]\n",
      " [5.00000e+01 4.00000e+04 9.00000e+01]\n",
      " [7.60000e+01 4.30000e+04 4.30000e+01]\n",
      " [7.60000e+01 4.30000e+04 5.40000e+01]\n",
      " [4.30000e+01 8.80000e+04 8.80000e+01]\n",
      " [5.60000e+01 6.00000e+03 3.00000e+01]\n",
      " [6.70000e+01 5.73560e+04 1.72068e+05]\n",
      " [2.00000e+01 2.00000e+04 4.00000e+04]\n",
      " [4.80000e+01 5.74730e+04 2.87365e+05]\n",
      " [2.40000e+01 3.58680e+04 2.15208e+05]\n",
      " [2.10000e+01 2.50000e+04 7.50000e+04]\n",
      " [2.50000e+01 1.50000e+04 6.05000e+01]\n",
      " [3.00000e+01 2.50000e+04 8.00000e+01]\n",
      " [3.50000e+01 3.00000e+04 9.00000e+01]\n",
      " [4.00000e+01 2.00000e+04 7.00000e+01]\n",
      " [4.50000e+01 4.50000e+04 1.20000e+02]\n",
      " [5.00000e+01 1.80000e+04 6.50000e+01]\n",
      " [5.50000e+01 2.20000e+04 7.50000e+01]\n",
      " [6.00000e+01 2.80000e+04 8.50000e+01]\n",
      " [6.50000e+01 3.50000e+04 1.00000e+02]\n",
      " [7.00000e+01 2.10000e+04 6.80000e+01]\n",
      " [7.50000e+01 3.30000e+04 9.50000e+01]\n",
      " [8.00000e+01 4.00000e+04 1.10000e+02]\n",
      " [8.50000e+01 6.00000e+04 1.50000e+02]\n",
      " [9.00000e+01 5.00000e+04 1.30000e+02]\n",
      " [9.50000e+01 7.00000e+04 1.40000e+02]\n",
      " [5.80000e+01 8.00000e+04 1.60000e+02]\n",
      " [2.20000e+01 1.20000e+04 4.50000e+01]\n",
      " [2.70000e+01 1.70000e+04 5.50000e+01]\n",
      " [3.30000e+01 1.90000e+04 7.00000e+01]\n",
      " [3.80000e+01 2.60000e+04 8.50000e+01]\n",
      " [4.30000e+01 3.00000e+04 9.50000e+01]\n",
      " [4.80000e+01 2.20000e+04 7.50000e+01]\n",
      " [5.30000e+01 3.20000e+04 1.10000e+02]\n",
      " [5.80000e+01 2.80000e+04 9.00000e+01]\n",
      " [6.30000e+01 2.50000e+04 8.00000e+01]\n",
      " [6.80000e+01 2.40000e+04 6.50000e+01]\n",
      " [7.30000e+01 1.90000e+04 5.00000e+01]\n",
      " [7.80000e+01 3.60000e+04 1.05000e+02]\n",
      " [8.30000e+01 3.10000e+04 9.50000e+01]\n",
      " [8.80000e+01 3.30000e+04 1.15000e+02]\n",
      " [9.30000e+01 3.70000e+04 1.20000e+02]\n",
      " [2.40000e+01 1.30000e+04 5.00000e+01]\n",
      " [2.90000e+01 2.10000e+04 7.00000e+01]\n",
      " [3.40000e+01 2.40000e+04 8.50000e+01]\n",
      " [3.90000e+01 3.10000e+04 9.00000e+01]\n",
      " [4.40000e+01 1.80000e+04 6.00000e+01]\n",
      " [4.90000e+01 3.00000e+04 1.00000e+02]\n",
      " [5.40000e+01 2.50000e+04 8.00000e+01]\n",
      " [5.90000e+01 2.90000e+04 7.00000e+01]\n",
      " [6.40000e+01 3.30000e+04 9.50000e+01]\n",
      " [6.90000e+01 3.20000e+04 9.00000e+01]\n",
      " [7.40000e+01 2.00000e+04 5.50000e+01]\n",
      " [7.90000e+01 3.40000e+04 1.00000e+02]\n",
      " [8.40000e+01 3.10000e+04 8.50000e+01]\n",
      " [8.90000e+01 3.80000e+04 1.15000e+02]\n",
      " [9.40000e+01 1.60000e+04 6.50000e+01]\n",
      " [2.60000e+01 1.40000e+04 5.80000e+01]\n",
      " [3.10000e+01 2.20000e+04 7.20000e+01]\n",
      " [3.60000e+01 2.70000e+04 8.80000e+01]\n",
      " [4.10000e+01 3.10000e+04 9.20000e+01]\n",
      " [4.60000e+01 3.40000e+04 1.10000e+02]\n",
      " [5.10000e+01 2.60000e+04 7.70000e+01]\n",
      " [5.60000e+01 2.80000e+04 8.20000e+01]\n",
      " [6.10000e+01 3.00000e+04 9.60000e+01]\n",
      " [6.60000e+01 1.90000e+04 5.40000e+01]\n",
      " [7.10000e+01 2.00000e+04 6.00000e+01]\n",
      " [7.60000e+01 3.70000e+04 1.05000e+02]\n",
      " [8.10000e+01 3.20000e+04 9.00000e+01]\n",
      " [8.60000e+01 4.10000e+04 1.22000e+02]\n",
      " [9.10000e+01 1.50000e+04 6.20000e+01]\n",
      " [9.70000e+01 1.80000e+04 7.40000e+01]\n",
      " [2.30000e+01 1.30000e+04 5.20000e+01]\n",
      " [2.80000e+01 1.90000e+04 6.70000e+01]\n",
      " [3.20000e+01 2.60000e+04 8.70000e+01]\n",
      " [3.70000e+01 3.10000e+04 9.20000e+01]\n",
      " [4.20000e+01 2.00000e+04 6.10000e+01]\n",
      " [4.70000e+01 2.90000e+04 7.50000e+01]\n",
      " [5.20000e+01 3.10000e+04 8.80000e+01]\n",
      " [5.70000e+01 2.40000e+04 6.60000e+01]\n",
      " [6.20000e+01 3.30000e+04 9.40000e+01]\n",
      " [6.70000e+01 3.50000e+04 1.01000e+02]\n",
      " [7.20000e+01 2.10000e+04 5.90000e+01]\n",
      " [7.70000e+01 2.80000e+04 7.90000e+01]\n",
      " [8.20000e+01 3.00000e+04 8.50000e+01]\n",
      " [8.70000e+01 4.20000e+04 1.25000e+02]\n",
      " [9.20000e+01 1.60000e+04 6.30000e+01]\n",
      " [9.60000e+01 1.90000e+04 7.10000e+01]\n",
      " [9.90000e+01 3.40000e+04 9.90000e+01]\n",
      " [2.50000e+01 1.50000e+04 6.05000e+01]\n",
      " [3.00000e+01 2.50000e+04 8.00000e+01]\n",
      " [3.50000e+01 3.00000e+04 9.00000e+01]\n",
      " [4.00000e+01 2.00000e+04 7.00000e+01]\n",
      " [4.50000e+01 4.50000e+04 1.20000e+02]\n",
      " [5.00000e+01 1.80000e+04 6.50000e+01]\n",
      " [5.50000e+01 2.20000e+04 7.50000e+01]\n",
      " [6.00000e+01 2.80000e+04 8.50000e+01]\n",
      " [6.50000e+01 3.50000e+04 1.00000e+02]\n",
      " [7.00000e+01 2.10000e+04 6.80000e+01]\n",
      " [7.50000e+01 3.30000e+04 9.50000e+01]\n",
      " [8.00000e+01 4.00000e+04 1.10000e+02]\n",
      " [8.50000e+01 6.00000e+04 1.50000e+02]\n",
      " [9.00000e+01 5.00000e+04 1.30000e+02]\n",
      " [9.50000e+01 7.00000e+04 1.40000e+02]\n",
      " [1.00000e+01 8.00000e+04 1.60000e+02]\n",
      " [2.20000e+01 1.20000e+04 4.50000e+01]\n",
      " [2.70000e+01 1.70000e+04 5.50000e+01]\n",
      " [3.30000e+01 1.90000e+04 7.00000e+01]\n",
      " [3.80000e+01 2.60000e+04 8.50000e+01]\n",
      " [4.30000e+01 3.00000e+04 9.50000e+01]\n",
      " [4.80000e+01 2.20000e+04 7.50000e+01]\n",
      " [5.30000e+01 3.20000e+04 1.10000e+02]\n",
      " [5.80000e+01 2.80000e+04 9.00000e+01]\n",
      " [6.30000e+01 2.50000e+04 8.00000e+01]\n",
      " [6.80000e+01 2.40000e+04 6.50000e+01]\n",
      " [7.30000e+01 1.90000e+04 5.00000e+01]\n",
      " [7.80000e+01 3.60000e+04 1.05000e+02]\n",
      " [8.30000e+01 3.10000e+04 9.50000e+01]\n",
      " [8.80000e+01 3.30000e+04 1.15000e+02]\n",
      " [9.30000e+01 3.70000e+04 1.20000e+02]\n",
      " [2.40000e+01 1.30000e+04 5.00000e+01]\n",
      " [2.90000e+01 2.10000e+04 7.00000e+01]\n",
      " [3.40000e+01 2.40000e+04 8.50000e+01]\n",
      " [3.90000e+01 3.10000e+04 9.00000e+01]\n",
      " [4.40000e+01 1.80000e+04 6.00000e+01]\n",
      " [4.90000e+01 3.00000e+04 1.00000e+02]\n",
      " [5.40000e+01 2.50000e+04 8.00000e+01]\n",
      " [5.90000e+01 2.90000e+04 7.00000e+01]\n",
      " [6.40000e+01 3.30000e+04 9.50000e+01]\n",
      " [6.90000e+01 3.20000e+04 9.00000e+01]\n",
      " [7.40000e+01 2.00000e+04 5.50000e+01]\n",
      " [7.90000e+01 3.40000e+04 1.00000e+02]\n",
      " [8.40000e+01 3.10000e+04 8.50000e+01]\n",
      " [8.90000e+01 3.80000e+04 1.15000e+02]\n",
      " [9.40000e+01 1.60000e+04 6.50000e+01]\n",
      " [2.60000e+01 1.40000e+04 5.80000e+01]\n",
      " [3.10000e+01 2.20000e+04 7.20000e+01]\n",
      " [3.60000e+01 2.70000e+04 8.80000e+01]\n",
      " [4.10000e+01 3.10000e+04 9.20000e+01]\n",
      " [4.60000e+01 3.40000e+04 1.10000e+02]\n",
      " [5.10000e+01 2.60000e+04 7.70000e+01]\n",
      " [5.60000e+01 2.80000e+04 8.20000e+01]\n",
      " [6.10000e+01 3.00000e+04 9.60000e+01]\n",
      " [6.60000e+01 1.90000e+04 5.40000e+01]\n",
      " [7.10000e+01 2.00000e+04 6.00000e+01]\n",
      " [7.60000e+01 3.70000e+04 1.05000e+02]\n",
      " [8.10000e+01 3.20000e+04 9.00000e+01]\n",
      " [8.60000e+01 4.10000e+04 1.22000e+02]\n",
      " [9.10000e+01 1.50000e+04 6.20000e+01]\n",
      " [9.70000e+01 1.80000e+04 7.40000e+01]\n",
      " [2.30000e+01 1.30000e+04 5.20000e+01]\n",
      " [2.80000e+01 1.90000e+04 6.70000e+01]\n",
      " [3.20000e+01 2.60000e+04 8.70000e+01]\n",
      " [3.70000e+01 3.10000e+04 9.20000e+01]\n",
      " [4.20000e+01 2.00000e+04 6.10000e+01]\n",
      " [4.70000e+01 2.90000e+04 7.50000e+01]\n",
      " [5.20000e+01 3.10000e+04 8.80000e+01]\n",
      " [5.70000e+01 2.40000e+04 6.60000e+01]\n",
      " [6.20000e+01 3.30000e+04 9.40000e+01]\n",
      " [6.70000e+01 3.50000e+04 1.01000e+02]\n",
      " [7.20000e+01 2.10000e+04 5.90000e+01]\n",
      " [7.70000e+01 2.80000e+04 7.90000e+01]\n",
      " [8.20000e+01 3.00000e+04 8.50000e+01]\n",
      " [8.70000e+01 4.20000e+04 1.25000e+02]\n",
      " [9.20000e+01 1.60000e+04 6.30000e+01]\n",
      " [9.60000e+01 1.90000e+04 7.10000e+01]\n",
      " [9.90000e+01 3.40000e+04 9.90000e+01]\n",
      " [2.00000e+01 1.10000e+04 4.00000e+01]\n",
      " [2.10000e+01 1.50000e+04 5.00000e+01]\n",
      " [2.20000e+01 1.40000e+04 4.50000e+01]\n",
      " [2.30000e+01 1.30000e+04 5.20000e+01]\n",
      " [2.40000e+01 1.20000e+04 4.80000e+01]\n",
      " [2.90000e+01 2.20000e+04 7.00000e+01]\n",
      " [3.20000e+01 2.10000e+04 7.50000e+01]\n",
      " [3.70000e+01 1.90000e+04 6.20000e+01]\n",
      " [4.10000e+01 2.00000e+04 6.50000e+01]\n",
      " [4.50000e+01 2.50000e+04 8.00000e+01]\n",
      " [5.10000e+01 2.70000e+04 8.80000e+01]\n",
      " [5.60000e+01 3.20000e+04 9.50000e+01]\n",
      " [6.10000e+01 3.40000e+04 1.05000e+02]\n",
      " [6.60000e+01 1.80000e+04 6.00000e+01]\n",
      " [7.10000e+01 3.00000e+04 9.00000e+01]\n",
      " [7.60000e+01 3.50000e+04 1.10000e+02]\n",
      " [8.10000e+01 4.00000e+04 1.15000e+02]\n",
      " [8.60000e+01 1.50000e+04 6.50000e+01]\n",
      " [9.10000e+01 2.30000e+04 8.50000e+01]\n",
      " [9.60000e+01 2.70000e+04 9.00000e+01]\n",
      " [1.10000e+01 2.90000e+04 9.50000e+01]\n",
      " [1.60000e+01 1.70000e+04 7.00000e+01]\n",
      " [1.90000e+01 2.50000e+04 8.00000e+01]\n",
      " [1.60000e+01 3.30000e+04 1.00000e+02]\n",
      " [2.10000e+01 3.60000e+04 1.20000e+02]\n",
      " [2.60000e+01 1.40000e+04 5.50000e+01]\n",
      " [3.10000e+01 1.80000e+04 6.00000e+01]\n",
      " [3.60000e+01 2.40000e+04 8.00000e+01]\n",
      " [4.10000e+01 3.00000e+04 9.00000e+01]\n",
      " [4.60000e+01 2.10000e+04 6.50000e+01]\n",
      " [5.10000e+01 1.70000e+04 5.50000e+01]\n",
      " [5.60000e+01 2.20000e+04 7.00000e+01]\n",
      " [6.10000e+01 3.20000e+04 1.00000e+02]\n",
      " [6.60000e+01 1.60000e+04 6.50000e+01]\n",
      " [7.10000e+01 2.60000e+04 8.00000e+01]\n",
      " [7.60000e+01 3.00000e+04 9.00000e+01]\n",
      " [8.10000e+01 3.30000e+04 1.00000e+02]\n",
      " [8.60000e+01 1.80000e+04 7.50000e+01]\n",
      " [9.10000e+01 2.50000e+04 8.50000e+01]\n",
      " [9.60000e+01 2.80000e+04 9.50000e+01]\n",
      " [2.70000e+01 3.00000e+04 1.20000e+05]\n",
      " [6.60000e+01 6.66660e+04 3.99996e+05]\n",
      " [4.50000e+01 1.03460e+04 8.27680e+04]\n",
      " [5.60000e+01 3.45660e+04 3.11094e+05]\n",
      " [7.00000e+01 4.63560e+04 2.31780e+05]\n",
      " [5.60000e+01 1.59570e+04 1.27656e+05]\n",
      " [2.70000e+01 7.77770e+04 5.44439e+05]\n",
      " [4.90000e+01 6.00000e+03 1.20000e+04]\n",
      " [4.90000e+01 4.00000e+04 1.60000e+05]\n",
      " [7.50000e+01 5.56765e+05 2.22706e+06]\n",
      " [5.50000e+01 6.00000e+03 1.80000e+04]\n",
      " [2.70000e+01 5.00000e+03 2.50000e+04]\n",
      " [5.80000e+01 1.50000e+04 9.00000e+04]\n",
      " [2.40000e+01 2.46000e+04 1.47600e+05]\n",
      " [8.90000e+01 2.00000e+04 4.00000e+04]\n",
      " [5.50000e+01 5.55550e+04 2.77775e+05]\n",
      " [5.50000e+01 5.55550e+04 2.77775e+05]]\n"
     ]
    }
   ],
   "source": [
    "print(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "LUHCVZWVNVb5"
   },
   "source": [
    "Choosing the number of clusters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "customer_data.fillna(customer_data.mean(), inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "25tFwgnZNiRF"
   },
   "source": [
    "WCSS  ->  Within Clusters Sum of Squares"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = customer_data[['Age', 'Average Amount Spent', 'Spending Score']].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "id": "LywIm4NDNIG1"
   },
   "outputs": [],
   "source": [
    "# finding wcss value for different number of clusters\n",
    "\n",
    "wcss = []\n",
    "for i in range(1, 11):\n",
    "    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)\n",
    "    kmeans.fit(X)\n",
    "    wcss.append(kmeans.inertia_)\n",
    " \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 301
    },
    "id": "1rWLIgxJPXI_",
    "outputId": "fabdc714-dcc6-465d-d84d-fc8d65b8c5dc"
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAi4AAAHJCAYAAACi47J4AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABOzElEQVR4nO3deVhUZf8G8PvMDMM+7IuCqIkioiCoKImKZOqbWj9tsTK3NPd8S83lda00W7Q0UbNy683KUtPKljfbNDO3yEotU0kBRUCWYYdZfn/AjIyADjAzZ5b7c11cMGfO8h0ey9vnec5zBK1WqwURERGRDZCIXQARERGRsRhciIiIyGYwuBAREZHNYHAhIiIim8HgQkRERDaDwYWIiIhsBoMLERER2QwGFyIiIrIZDC5ERI3ANTuJxMXgQmQG8+fPR0RExC2/Ro8eDQAYPXq0/mdT27Nnz23ruHDhgr7m5ORk/bHJycmYP3++WepqrPp+n1FRUUhMTMQzzzyDq1evNup8ut9LRkaG0ccolUrMnTsXJ06cMGr/n376Cf/+97+RlJSEzp07o1evXpg8eTIOHTrUqFqbKyMjAxEREdizZ49Fr0tkLjKxCyCyR9OmTcPDDz+sf71hwwacOXMGKSkp+m0eHh4WqyclJQUBAQH1vhcaGmqxOpojICDA4PenUqmQlpaGVatWITU1FZ999hlcXFyMOldSUhJ27tyJwMBAo69/9uxZ7Nu3D/fff/9t9125ciW2bduGu+++G8888wyCgoKQk5ODffv2YeLEiZg/fz7Gjx9v9LWJ6AYGFyIzCAsLQ1hYmP61r68v5HI5unbtKko9kZGRNhNQGlLf76979+5wcnLCvHnz8M0332DIkCFGncvX1xe+vr5mqBL4+OOPsW3btnrDyb/+9S88//zzWL16NQYPHowWLVqYpQYie8ahIiIroNVq8dZbbyEpKQnR0dEYOXIkfvvtN4N9zp07h8mTJyMuLg5xcXGYPn060tPTzVpXVVUVli9fjh49eqB79+6YN28e8vLyDPY5fPgwHn30UXTr1g09e/bE7Nmz9UM333zzDSIiInDmzBn9/nv37kVERAQ++ugj/bazZ88iIiICqampja6xS5cuAIDMzEyjagLqDhXNnz8f48aNw+7duzFo0CB07twZ9913Hw4ePAgAOHr0KMaMGQMAGDNmzC2H9tavX4/o6GiMGzeu3venT5+OxMRE5Ofn688dERGBDz74AP3790dcXBwOHz4MAPjoo48wYsQIdO3aFdHR0bjvvvvwxRdf1Pkcp06dwvDhwxEdHY1hw4bhyy+/rHPdnJwczJw5E7GxsYiPj8fixYtRUlJy298vkbVhcCGyAidPnsTXX3+NxYsX45VXXkF2djamTp0KlUoFAEhLS8PDDz+M69ev46WXXsKKFSuQnp6ORx55BNevX7/t+TUaDVQqVZ0vjUZzy+O++OILnD59Gi+++CLmzZuH77//Hk888QTUajWA6hDy+OOPo0WLFnj11VexYMECpKamYuTIkbh+/ToSEhIgl8vx008/6c/5888/A4DBXJGDBw/C19cXMTExjf7dpaWlAYC+h+t2NTXkjz/+wObNmzFz5kysX78eUqkUTz75JAoLCxEVFYUlS5YAAJYsWYKlS5fWe44///wT6enpGDJkCARBqHcfX19fvPHGG+jUqZPB9pSUFMybNw9LlixBbGwsduzYgSVLlmDAgAHYtGkTVq1aBblcjjlz5iArK8vg2MmTJ+Ouu+5CSkoK2rZti6eeego//PCDwT5r165FixYtsGHDBowdOxYffvihwdAbka1wiKGiTZs24ccff8R///tfkx7/7bffYv369bh48SJ8fHwwaNAg/Pvf/zZ6nJ1IRy6X480334S3tzeA6omgixYtwvnz59GxY0ekpKTA1dUV27Zt08+NSUhIwIABA/D2229j3rx5tzz/3XffXe/2pKQkbNq0qcHjfHx8sHnzZri5uelfT58+HQcPHkS/fv2watUqJCYmYvXq1fpj4uLicM8992Dz5s2YO3cu4uPjceTIEUycOBEAcOTIEURFReH48eP6Yw4dOoR+/fpBIrn1v6V0QQ4AiouL8fvvv2PlypUIDQ1FUlISNBqNUTXVp6ioCHv27NEHIDc3Nzz22GP4+eefMWjQIISHhwMAwsPD9T/fTNcD1qZNG4PtWq1WH/Z0JBKJwed99NFHMXjwYINzTZgwAdOmTdNvCwkJwYgRI3Dy5EmDYbHRo0dj+vTpAIA+ffpg+PDhWL9+Pfr166ffZ9CgQViwYAGA6j87hw8f1odIIlti98Flx44dWLNmDbp3727S40+cOIEZM2Zg5syZGDx4MC5duoQlS5agoKAAK1euNEXp5EDCw8P1oQW4MWG2qKgIQHUvRXx8PFxcXPR/eXt4eKB79+4GvRkN2bhxY72TcxUKxS2P69evnz60ANV3GslkMhw/fhxhYWHIycnB7NmzDY4JCwtDbGwsjh07BqA6HK1evRqVlZXIzMxEVlYWFi5ciCeffBKZmZnw8vJCamoqHnvssVvWkpmZiaioqDrbY2Ji8Nxzz8HFxQUXLlwwqqb6+Pr6GsxLCg4OBgCUlZXdsq7aGurB2rVrFxYtWmSwbfjw4XjxxRf1ryMjIw3e193RpVQqcfHiRVy6dAlHjx4FAFRWVtY5l44gCLj77ruxbt06lJeX67ff/P+w0NBQnDx50tiPRmQ17Da4XLt2DUuXLsXRo0fr/OvHFMd/8MEH6NmzJ6ZMmQKg+l9YTz/9NBYtWoRnn30Wcrm8mZ+AHEntcABA/y9x3V+EBQUF+Pzzz/H555/XOdaYSaYdOnRo0uTcm8OORCKBj48PlEolCgoKAAD+/v51jvP399fPa0lKSsLy5cvxyy+/4OLFi2jbti369+8PNzc3HD9+HG5ubhAEAYmJibetZePGjfrXcrkcwcHB8PLy0m8ztqb6uLq6GrzWDfXcbjittpYtWwIwnG8DAHfddRc6duyofz116tQ6x978Z+Dy5ctYsmQJjhw5AicnJ9xxxx36c9y8lszNd0f5+flBq9VCqVTqt938+SQSCdekIZtkt8Hl9OnTcHJywieffIL169fX+R/Jd999h3Xr1uH8+fMICgrCkCFDMG3aNH3guN3xjz/+eJ1ubYlEgqqqKhQXF5vtjgVyTJ6enrjzzjvrvYVWJjPff8a6IKCjVquRn58PPz8/fQ9Rbm5uneNycnLg4+MDAGjVqhXuuOMOHDlyBGlpaYiPj4eTkxPi4uJw9OhRSKVS9OjR47a3h8vlcv1E3IYYW5O5REVFISgoCF9++SVGjRql337zXUy3+4eNRqPBpEmT4OTkhF27diEyMhIymQznz5/Hvn376uxfUFBgENZyc3MhlUrh7e2N7OxsE3wyIutht5Nzk5OTsW7dOrRq1arOewcPHsRTTz2Fhx56CJ999hmWLl2KL774As8884xRxwNAp06dDP4FVVVVhW3btqFz584MLWRy8fHxOH/+PCIjI9GlSxd06dIFnTt3xrZt2/D111+b7bqHDx82mFfy1VdfQaVSoWfPnmjbti0CAgLw2WefGRyTnp6OX3/9FXFxcfptSUlJOHr0KE6ePImePXsCAHr27ImjR4/i0KFD6N+/v0nqbUxNjSWVSm+7j0QiwYwZM3Ds2DFs37693n2uXr2K4uLiW54nPz8faWlpeOCBB9ClSxd9ONXd5XRzL9CBAwf0P2u1Wvzvf/9Dt27d2PNLdslue1xu5Y033sBDDz2kXyAsLCwMzz77LMaOHYuMjIxGd6mrVCrMnTsXf//9N3bs2GGOksnB6Ra0mzx5Mh555BE4Oztj586dOHDgAF5//fXbHn/27Nl6eyGA6gmfDS1Ol5OTgyeffBKjR4/GP//8g1dffRW9e/dGQkICBEHArFmzsGDBAsyePRv33nsv8vPzkZKSAi8vL4PeoX79+mHLli0AqkMYAPTq1Us/gdZUwUUikRhdU2N5enoCAL7//nt4eXkZ/MOltoceeggZGRlYuXIlDh48iKFDhyIkJASFhYX48ccfsW/fPjg5Od3yM/v5+SEkJAQ7duxAcHAwFAoFDh06hHfeeQdA3Xk3L7/8MioqKtC2bVt89NFHuHDhQoPBicjWOWRwOXPmDH777Tfs2rVLv0031nvhwoVGBZfi4mI89dRTOHbsGFJSUhAdHW3yeok6duyIHTt24LXXXsPcuXOh1WrRoUMHrF+/Hnfddddtj58xY0aD7y1YsKDBNUceffRRFBUVYfr06ZDL5Rg2bBieeeYZ/fyPESNGwN3dHZs2bcL06dPh4eGBPn36YNasWQZhqFu3bvD09IS/v79+e1RUFDw8PBAUFNRgz2ZTGFtTY7Vv3x5Dhw7Fjh07cOjQoTq9OrXNmjULycnJ+OCDD5CSkoLs7Gy4uLggPDwcM2bMwAMPPGAwGbs+GzZswIoVKzB//nzI5XKEh4dj48aNeOGFF3DixAmDtWSWLVuGTZs2IT09HZ06dcKWLVuafEMCkbUTtA4wO2v+/PnIzMzU384cHR2Nxx9/3GAmvk5AQECdSXI3H6+TnZ2NJ554ApmZmdi4cSN69Ohhvg9BRHSTPXv2YMGCBfjmm29sfmVkImPZ7RyXW2nfvj3S0tLQunVr/VdWVhZefvllo1eSLCwsxNixY5GXl4cdO3YwtBAREVmAQwaXJ554Al999RVSUlKQlpaGI0eOYMGCBSgqKjK6K3nlypVIT0/HK6+8Al9fX+Tk5Oi/bl5oioiIiEzDIee4DB48GK+99ho2bdqEN954A97e3khOTsacOXOMOl6tVuPzzz9HVVUVxo4dW+d9dtsSkSWMGDECI0aMELsMIotyiDkuREREZB8ccqiIiIiIbBODCxEREdkMBhciIiKyGXY3OVer1UKj4bSdW5FIBP6OrAjbw7qwPawP28S6mKM9JBJBv7Dl7dhdcNFotMjLM24tFkckk0ng4+MOpbIUKpXxT70l82B7WBe2h/Vhm1gXc7WHr687pFLjgguHioiIiMhmMLgQERGRzWBwISIiIpvB4EJEREQ2g8GFiIiIbAaDCxEREdkMBhciIiKyGQwuREREZDMYXIiIiMhmMLgQERGRzWBwISIiIpvB4EJEREQ2g8GFiIiIbAaDi5EOnrqCU+dzxS6DiIjIoTG4GKGsQoVtX/yJjfv+gErNx6oTERGJhcHFCM5yKZzlUlRWaXAtv0zscoiIiBwWg4sRJIKAUH93AEBmTrHI1RARETkuBhcjhQZ6AADSsxlciIiIxGIVwWXv3r2455570KVLFwwZMgRffPGF2CXVERpQHVwyGFyIiIhEI3pw2bdvHxYuXIhRo0Zh//79GDp0KGbNmoXU1FSxSzMQGlA9VJSRUyJyJURERI5L1OCi1Wqxdu1ajBkzBqNGjUJYWBimTp2KO++8E8eOHROztDp0Q0XXleUoLVeJXA0REZFjkol58bS0NGRmZmLYsGEG2zdv3ixSRQ1zd3GCj6cz8osqkJlbjPah3mKXRERE5HBEDy4AUFpaigkTJuDMmTMIDQ3F1KlTkZyc3OTzymTm6UgKC/JAflEFrlwvRWQbX7Ncw9ykUonBdxIX28O6sD2sD9vEulhDe4gaXIqLqye6zps3DzNmzMCcOXPw1VdfYdq0adi6dSsSEhIafU6JRICPj7upSwUAhLfywanz15FdWG62a1iKQuEqdglUC9vDurA9rA/bxLqI2R6iBhcnJycAwIQJEzB8+HAAQGRkJM6cOdPk4KLRaKFUlpq0Tp0ALxcAwPnL+cjPt81JulKpBAqFK5TKMqi5CrDo2B7Whe1hfdgm1sVc7aFQuBrdiyNqcAkKCgIAdOjQwWB7eHg4vv/++yafV6Uyzx/ulr5uAICMnGJUVakhCIJZrmMJarXGbL8najy2h3Vhe1gftol1EbM9RB00jIqKgru7O06dOmWw/dy5cwgLCxOpqoYF+7lBKhFQVqHGdWW52OUQERE5HFF7XFxcXDBx4kSsX78eQUFBiI6Oxv79+3H48GFs27ZNzNLqJZNK0MLPDRk5JcjIKYG/F8dciYiILEnU4AIA06ZNg6urK1577TVcu3YN7dq1w7p169CzZ0+xS6tXaKBHdXDJLkbXcH+xyyEiInIoogcXABg/fjzGjx8vdhlGqV76/xoy+LBFIiIii+ON8Y2kf2YRl/4nIiKyOAaXRtI9syjreimqOMOdiIjIohhcGsnH0xnuLjJotFpcvc5eFyIiIkticGkkQRAQUjNclJ7NeS5ERESWxODSBK1qgksm57kQERFZFINLE4QGVs9zSeedRURERBbF4NIEN+4sYnAhIiKyJAaXJgipubOosLgSRaWVIldDRETkOBhcmsBFLkOAd/WTormeCxERkeUwuDSRfriIdxYRERFZDINLE3GeCxERkeUxuDRRq0AGFyIiIktjcGki3QTdzNwSaDRakashIiJyDAwuTRTk4wYnmQSVVRrkFJSJXQ4REZFDYHBpIolEQEv/moXoOEGXiIjIIhhcmqEVJ+gSERFZFINLM4TWzHPhWi5ERESWweDSDKG8s4iIiMiiGFyaQRdccvLLUFGpFrkaIiIi+8fg0gwKNzkU7nJoUX1bNBEREZkXg0sztdLPc+FwERERkbkxuDRTCJ9ZREREZDEMLs3Epf+JiIgsh8GlmW48bLEEWi2X/iciIjInBpdmaunvBkEAisuqUFBcKXY5REREdo3BpZmcZFIE+7oBADI5XERERGRWDC4moBsuSmdwISIiMisGFxPQL/3PO4uIiIjMisHFBG4s/c9F6IiIiMyJwcUEdENFV3JLoFJrRK6GiIjIfjG4mICflwtc5FKoNVpcyysVuxwiIiK7xeBiAhJB4ARdIiIiC2BwMRHdBN1MznMhIiIyGwYXE9FN0E3nnUVERERmw+BiIrqhIi5CR0REZD4MLiaiGyq6rqxAaXmVyNUQERHZJwYXE3FzcYKvwhkA13MhIiIyFwYXE7rxpGgOFxEREZkDg4sJ3Qgu7HEhIiIyBwYXEwoN5DOLiIiIzInBxYRqDxVptVqRqyEiIrI/DC4mFOzrBqlEQHmlGtcLy8Uuh4iIyO4wuJiQTCpBC7/q4SIu/U9ERGR6ogeXa9euISIios7Xnj17xC6tSVrp5rlwgi4REZHJycQu4M8//4SzszMOHDgAQRD02z09PUWsqulCAz2A09c4QZeIiMgMRA8u586dQ5s2bRAYGCh2KSbBtVyIiIjMR/Shor/++gvt2rUTuwyT0QWXa3llqFKpRa6GiIjIvlhFj4uPjw9GjRqFtLQ0tG7dGlOnTkXfvn2bfE6ZTLw85u/tAndXJ5SUVeFaQRnaBCtEq6U+UqnE4DuJi+1hXdge1odtYl2soT1EDS4qlQoXL15EeHg45s+fDw8PD+zfvx+TJk3C1q1bkZCQ0OhzSiQCfHzczVCt8e5o6YXfL+Qir7gKsSLX0hCFwlXsEqgWtod1YXtYH7aJdRGzPUQNLjKZDEePHoVUKoWLiwsAoHPnzvj777+xefPmJgUXjUYLpbLU1KU2SrCvK36/APz1z3XEhfuJWsvNpFIJFApXKJVlUKs1Ypfj8Nge1oXtYX3YJtbFXO2hULga3Ysj+lCRu3vdHon27dvjxx9/bPI5VSpx/3CH+Fd/pstZRaLX0hC1WmO1tTkitod1YXtYH7aJdRGzPUQdNPz7778RFxeHo0ePGmz/448/EB4eLlJVzRcSwLVciIiIzEHU4NKuXTvccccdeO6553DixAlcuHABK1euxK+//oqpU6eKWVqzhPi7QwBQWFIJZWml2OUQERHZDVGDi0QiwRtvvIHo6Gg89dRTGD58OE6dOoWtW7eiQ4cOYpbWLC5yGQK8qycuZXIhOiIiIpMRfY6Lv78/Vq5cKXYZJhca6IHsgjKk55Qgso2v2OUQERHZBd4Ybyahunku7HEhIiIyGQYXM+HS/0RERKbH4GImrQKrg0tmbgk0Gq3I1RAREdkHBhczCfB2hVwmQZVKg+yCMrHLISIisgsMLmYikQg31nPhPBciIiKTYHAxoxDOcyEiIjIpBhczalUTXNLZ40JERGQSDC5mpLslOpNL/xMREZkEg4sZhdTcWZRdUIbySpXI1RAREdk+BhczUrjJ4eUuB1B9WzQRERE1D4OLmYXW9LrwziIiIqLmY3AxM/3S/5znQkRE1GwMLmamX/qfPS5ERETNxuBiZrWfWaTVcul/IiKi5mBwMbOW/m6QCAJKylUoKK4UuxwiIiKbxuBiZk4yKYL93ABwIToiIqLmYnCxgBsL0TG4EBERNQeDiwXo5rmkM7gQERE1C4OLBdy4s4i3RBMRETUHg4sFhAZWDxVdvV4ClVojcjVERES2i8HFAvwULnB1lkKt0SIrr1TscoiIiGwWg4sFCIKAEC5ER0RE1GwMLhZyYyE6znMhIiJqKgYXC2mlf2YRe1yIiIiaisHFQkJqLf1PRERETcPgYiG6oaI8ZQVKyqtEroaIiMg2MbhYiJuLDH4KZwCcoEtERNRUDC4WxAm6REREzcPgYkGhgZznQkRE1BwMLhYUygm6REREzcLgYkE3elxKoNFqRa6GiIjI9jC4WFCQjytkUgEVlWpcLywXuxwiIiKbw+BiQTKpBC39ahai451FREREjcbgYmFciI6IiKjpGFwsrFXNPJd03hJNRETUaAwuFhZa88yiTPa4EBERNRqDi4Xp7izKyitFZZVa5GqIiIhsC4OLhXm5y+Hh6gStFrh6vVTscoiIiGwKg4uFCYKgHy5K551FREREjcLgIgKuoEtERNQ0DC4i4DOLiIiImobBRQS6W6K5CB0REVHjMLiIoKW/OwQAytIqFJZUil0OERGRzbCq4JKWlobY2Fjs2bNH7FLMytlJikAfVwAcLiIiImoMqwkuVVVVmDNnDkpLHeMWYd0E3UwOFxERERnNaoLLunXr4OHhIXYZFhOqX/qfwYWIiMhYMrELAIDjx49j586d2Lt3L5KSkpp9PpnMavJYg8KCPQEAmbklFq1XKpUYfCdxsT2sC9vD+rBNrIs1tIfowUWpVGLu3LlYtGgRWrRo0ezzSSQCfHzcTVCZeXVuHwAAuJJTAoWXG6QSwaLXVyhcLXo9ujW2h3Vhe1gftol1EbM9RA8uy5YtQ2xsLIYNG2aS82k0WiiV1j9PxlkCyJ0kqKzS4K+LOWjhZ5mwJZVKoFC4Qqksg1qtscg1qWFsD+vC9rA+bBPrYq72UChcje7FETW47N27FydOnMCnn35q0vOqVLbxhzvE3wNpV5X452oRArwsm17Vao3N/J4cAdvDurA9rA/bxLqI2R6iDhru3r0b169fR1JSEmJjYxEbGwsAWLp0KSZOnChmaRahe2YRF6IjIiIyjqg9LqtWrUJ5ebnBtoEDB2LmzJm49957RarKcrj0PxERUeOIGlyCgoLq3e7n59fge/ZEt5YLnxJNRERkHN5fJiLdUFFuYTnKKlQiV0NERGT9RL+r6GZ//fWX2CVYjKebHN4echQUVyIztwThIV5il0RERGTV2OMiMt1wEee5EBER3R6Di8j0E3Q5z4WIiOi2GFxEpr8lOqdE5EqIiIisH4OLyPRDRdnF0Gq1IldDRERk3RhcRNbCzx0SQUBphQr5RRVil0NERGTVGFxE5iSToIWfGwBO0CUiIrodBhcrEMJ5LkREREZhcLECrXhnERERkVEYXKxACNdyISIiMgqDixVoVRNcrl4vhUrNx7YTERE1hMHFCvgqnOHqLINao0XW9VKxyyEiIrJaDC5WQBAE/UJ06RwuIiIiahCDi5Xg0v9ERES3x+BiJW48bJG3RBMRETWEwcVKtOKdRURERLfF4GIldIvQ5RdVoLisSuRqiIiIrBODi5VwdZbB38sFAJDJXhciIqJ6MbhYEc5zISIiujUGFysSGlhzSzTvLCIiIqoXg4sV0fW4cKiIiIiofiYJLnl5eaY4jcOrPVSk0WpFroaIiMj6GB1c0tPT8fzzz+Obb77Rbztw4AASExPRu3dv9OnTB59//rlZinQUQb6ukEklqKhSI7ewXOxyiIiIrI7MmJ3S09Px4IMPoqKiAp06dQIApKWl4amnnoKvry/mz5+PixcvYs6cOQgMDET37t3NWrS9kkokaOnvhsvXipGRXYxAb1exSyIiIrIqRgWXN954A76+vti+fTsCAgIAAFu3boVarcaqVasQHx8PAKisrMRbb73F4NIMoQEe1cElpxhxHQLELoeIiMiqGDVU9NNPP2HChAn60AIABw8eRGBgoD60AMDAgQNx6tQp01fpQPTzXHhnERERUR1GBZfc3FyEhYXpX6enpyMrKws9e/Y02M/T0xMlJVyDpDla1TxsMZ1ruRAREdVhVHBxd3eHUqnUvz527BgEQUCvXr0M9ktPT4e3t7dJC3Q0oTVL/2fnl6KiSi1yNURERNbFqODStWtXgzuG9u3bB6lUin79+um3abVafPjhh4iOjjZ9lQ5E4S6Hp5sTtFrgSi57XYiIiGozanLuE088gbFjxyIrKwsajQapqakYOXIk/Pz8AABHjhzB9u3b8euvv2Lr1q1mLdjeCYKA0AAPnL2Uj4ycYrRtoRC7JCIiIqthVI9Lt27d8NZbb8HJyQlFRUWYOHEiFi1apH9/zpw5OHr0KJYtW1Zn+Iga78YEXfa4EBER1WZUjwsAJCQkICEhod73Nm7ciDZt2kChYO+AKejmuWRw6X8iIiIDRgeXW+G8FtMKDdQt/c/gQkREVJvRS/4XFRVhy5YtOHbsmH7bqVOn8MADDyA2NhYjR47EyZMnzVKko2np7w4BQFFpFQpLKsUuh4iIyGoYFVzy8vIwYsQIvPLKKzh79iwA4Nq1axg/fjzS0tLw4IMPQqFQYPz48Th37pxZC3YEzk5SBPq6AeBCdERERLUZveR/ZWUlPv74Y3Ts2BEAsG3bNpSVlWHdunUYMGAAAGDatGnYsGED1qxZY7aCHUVogDuu5ZUiI6cYUW19xS6HiIjIKhjV4/L9999j0qRJ+tACAN988w28vb31oQUA/u///g8nTpwwfZUOqBWX/iciIqrDqOCSlZWF9u3b619nZ2fj8uXLBs8pAgBfX18UFhaatkIHFaILLlz6n4iISM+o4OLs7IyysjL96+PHjwNAnTVbrl27Bk9PTxOW57haBVbfEp2ZWwK1RiNyNURERNbBqOASFRWFgwcP6l9/8cUXkEgkBkv+A8Ann3yCyMhI01booPy9XeHsJIVKrcG1vLLbH0BEROQAjJqcO2bMGEyfPh1FRUVQq9U4cOAABg0ahJYtWwIALl26hO3bt+PgwYOcmGsiEkFASIA7Ll5RIiOnGC393cUuiYiISHRG9bgkJyfjhRdeQGpqKr799lv861//wooVK/TvP/zww3j//fcxadIkDBo0yGzFOhr90v9ciI6IiAhAI1bOHT58OIYPH17ve88++yzat2+Ptm3bmqwwqrX0P59ZREREBKARK+feSnJycpNDy/Xr1/HMM8+gV69eiI2NxaRJk3DhwgVTlGXzWnHpfyIiIgNGB5fi4mK89NJL2LVrl8H2yspKJCUlYfny5QZ3Hhlr+vTpuHTpEt58803s2rULLi4uGDduXJPOZW90t0TnFpajrEIlcjVERETiMyq4lJSUYOzYsdi2bRtyc3MN3isuLkZ0dDQ++OADjBs3DuXl5UZfvLCwECEhIVi+fDmio6PRrl07TJs2DdnZ2fj7778b90nskIerE3w8nQEAmVzPhYiIyLjg8s477+Dy5cvYsWMHpkyZYvCer68vNmzYgLfffhvnzp3Du+++a/TFvby8sHr1anTo0AFA9TORtm3bhuDgYISHhzfiY9ivEN08Fw4XERERGTc59/PPP8fEiRMRFxfX4D69evXCY489hs8++wwTJ05sdCGLFy/Ghx9+CLlcjo0bN8LNza3R59CRyUwydccqtA7yxB8X85CZW2KSzyWVSgy+k7jYHtaF7WF92CbWxRraw6jgkpGRgZiYmNvuFx8fjx07djSpkLFjx2LkyJHYsWMHpk+fjvfeew9RUVGNPo9EIsDHx37WPIlo64f9Ry4hK7/MpJ9LoXA12bmo+dge1oXtYX3YJtZFzPYwKri4ubmhpOT2cyw0Gg2cnZ2bVIhuaGjFihU4deoU3n33XaxcubLR59FotFAqS5tUgzXydXcCAKRlFiIvrxiCIDTrfFKpBAqFK5TKMqjVfJSA2Nge1oXtYX3YJtbFXO2hULga3YtjVHCJjIzEwYMHcdddd91yvx9++AGtW7c26sJA9ZyWI0eOYNCgQZDJqkuRSCQIDw9Hdna20ee5mUplP3+4A71dIZUIKK1QISe/DL4KF5OcV63W2NXvydaxPawL28P6sE2si5jtYVS8efDBB7F792588803De7z3Xff4cMPP8R9991n9MVzc3Mxa9YsHDlyRL+tqqoKZ86cQbt27Yw+jz2TSSUI9que75OezQm6RETk2IzqcRk0aBD+97//YcaMGejXrx+SkpIQGhoKtVqNK1eu4IcffsAPP/yAfv36YeTIkUZfvEOHDujbty+WL1+O5cuXw8vLC5s2bYJSqcS4ceOa+pnsTqsAD2TmlCAjpxgx4f5il0NERCQao5f8X7VqFSIiIrB161Z8//33+rkWWq0W/v7+mD17NsaNGweJpHEzjV999VWsXr0aTz/9NIqKitC9e3fs2LFD/wBHqn1LNNdyISIixyZotVrt7Xb65JNPkJiYCF9fX6hUKpw+fRpXr16FTCZDy5YtERkZ2exJo6aiVmuQl2dff8H/diEXaz76DSH+7nh+Ys9mnUsmk8DHxx35+SUcL7YCbA/rwvawPmwT62Ku9vD1dTft5Ny5c+dCEAR06NABvXv3RmJiIpKTkyGXy5tVKBlH95TorLxSVKk0cLKjdWqIiIgaw6jgsnv3bhw/fhwnTpzAxx9/jC1btsDZ2RlxcXHo3bs3evfujcjISHPX6rB8PJ3h5ixDaYUKV6+XICzIU+ySiIiIRGFUcImKikJUVJR+wuyFCxdw7NgxnDx5Ejt27MCqVavg6+uLhIQEJCYmYvjw4eas2eEIgoDQAHecyyhEZg6DCxEROS6jJ+fW1q5dO7Rr1w6PPPIIAODo0aN477338NVXX+Hzzz9ncDGD0EAPnMsoRHpOMRLELoaIiEgkTQoueXl5OHToEI4cOYKjR48iKysLbm5u6NOnDxITE01dI+HGPBc+bJGIiByZUcFFrVYjNTUVhw4dwqFDh/Dnn38CqB5Cuu+++5CYmIiuXbvqV78l0wsNrAkuXISOiIgcmFFJo2fPnigpKUGLFi2QkJCAJ554AnfeeSe8vLzMXR/VCPGvXsuloLgSxWVV8HB1ErkiIiIiyzPqvtri4mJ4eXnpV83t06cPQ4uFuTrL4O9V/Zwi9roQEZGjMqrHZdeuXTh06BB+/PFHfPTRRwCA6OhoJCYmIjExEdHR0WYtkqqFBnggt7AcGTnF6NjaR+xyiIiILM6o4NK5c2d07twZU6dORXFxMX766Sf8+OOP2LVrF15//XV4e3vjzjvvRGJiInr37o2goCBz1+2QQgM98Ov5XE7QJSIih9Xo2bQeHh4YOHAgBg4cCKB6TZeff/4ZR48exbJly6BSqXDmzBmTF0pAq5oJuunZ9vVIAyIiImM1+TaggoICpKam4pdffsGvv/6K06dPQ6PRcNjIjEJrHraYmVsMjVYLiZU8H4qIiMhSjA4u//zzD3755Rf9V1paGrRaLdq3b4+EhARMmDABPXr0gLu7uznrdWiBPq5wkklQWaVBTkEZgnzcxC6JiIjIoowKLr169UJhYSG0Wi1atmyJhIQETJs2DQkJCfDz8zN3jVRDKpGgpZ87Ll0rQkZ2CYMLERE5HKPXcbnzzjuRkJCAsLAwc9dEtxAaWBNccorRLSJA7HKIiIgsyqjgsnbtWnPXQUbi0v9EROTIjFqAjqwHl/4nIiJHxuBiY3Q9Ltn5ZaioUotcDRERkWUxuNgYL3c5FG5O0AK4ksv1XIiIyLEwuNigkAAOFxERkWNicLFB+hV0OUGXiIgcDIOLDQrRraCbw6EiIiJyLAwuNujGM4uKodVqRa6GiIjIchhcbFBLP3cIAlBcVoXCkkqxyyEiIrIYBhcbJHeS6pf750J0RETkSBhcbNSNheg4z4WIiBwHg4uNCq2ZoMseFyIiciQMLjaqFddyISIiB8TgYqNCaoaKrlwvgVqjEbkaIiIiy2BwsVH+Xi5wlkuhUmuRlVcmdjlEREQWweBioySCgFB/3UJ0HC4iIiLHwOBiw0JrLURHRETkCBhcbFhozQRdLv1PRESOgsHFhuluiWaPCxEROQoGFxumGyq6rixHablK5GqIiIjMj8HFhrm7OMHH0xkAkJnLXhciIrJ/DC42Tvek6AzOcyEiIgfA4GLjQnRL/3OeCxEROQAGFxunW/o/nWu5EBGRA2BwsXE3bokuhlarFbkaIiIi82JwsXHBfm6QSgSUVahxXVkudjlERERmxeBi42RSCVr4uQHgBF0iIrJ/DC52QLeeCyfoEhGRvRM9uBQUFGDJkiXo27cv4uLi8Mgjj+DEiRNil2VTdPNcMjhBl4iI7JzowWXWrFlITU3Fq6++it27dyMyMhITJkzAxYsXxS7NZtwILhwqIiIi+yZqcLl06RIOHz6MZcuWoXv37mjbti0WL16MwMBAfPrpp2KWZlN0zyzKul6KKpVG5GqIiIjMR9Tg4uPjgzfffBNdunTRbxMEAYIgQKlUiliZbfHxdIa7iwwarRZXr7PXhYiI7JdMzIsrFAr069fPYNtXX32FS5cu4T//+U+TzyuTiT4CZnGhgR7463IBrl4vxR0hXg3uJ5VKDL6TuNge1oXtYX3YJtbFGtpD1OBys19++QULFizAwIEDkZSU1KRzSCQCfHzcTVuYDWjfygd/XS5AtrLCqM+vULhaoCoyFtvDurA9rA/bxLqI2R5WE1wOHDiAOXPmIC4uDqtWrWryeTQaLZTKUhNWZhsCvFwAAOcv5yM/v+HhIqlUAoXCFUplGdRqzocRG9vDurA9rA/bxLqYqz0UCleje3GsIri8++67WLFiBQYPHoyXXnoJcrm8WedTOeAE1ZY1i9Bdzi4y6vOr1RqH/D1ZK7aHdWF7WB+2iXURsz1EHzR877338Pzzz2PUqFF49dVXmx1aHJXuKdGFxZUoKq0UuRoiIiLzEDW4pKWl4YUXXsDdd9+NyZMnIzc3Fzk5OcjJyUFRUZGYpdkcF7kMAd7Vw0Vcz4WIiOyVqENFX331FaqqqvD111/j66+/Nnhv+PDhePHFF0WqzDaFBnggp6AcGdnFiGztI3Y5REREJidqcJkyZQqmTJkiZgl2JTTAA6l/53LpfyIisluiz3Eh02kVyGcWERGRfWNwsSO6CbqZuSXQaLQiV0NERGR6DC52JMjHDU4yCSqrNMgpKBO7HCIiIpNjcLEjEomAlv7VvS4cLiIiInvE4GJnWgVUz3NJz2ZwISIi+8PgYmdCdfNcuJYLERHZIQYXOxNac2dROoeKiIjIDjG42BldcMnJL0NFpVrkaoiIiEyLwcXOKNzkULjLoUX1bdFERET2hMHFDrUK4J1FRERknxhc7FBIzZ1FGbyziIiI7AyDix3i0v9ERGSvGFzsUKiuxyWnBFotl/4nIiL7weBih1r6u0EQgOKyKhQUV4pdDhERkckwuNghJ5kUwb5uAIBMDhcREZEdYXCxU7rhIi5ER0RE9oTBxU7plv7PyOZaLkREZD8YXOxUKO8sIiIiO8TgYqd0Q0VXr5dApdaIXA0REZFpMLjYKT8vF7jIpVCptbiWVyp2OURERCbB4GKnJIJgsJ4LERGRPWBwsWOhfGYRERHZGQYXO6aboJvOZxYREZGdYHCxY7qhIi5CR0RE9oLBxY7phoquKytQWl4lcjVERETNx+Bix9xcnOCrcAbACbpERGQfGFzs3I07izhcREREto/Bxc7xlmgiIrInDC52LjRQ98wi9rgQEZHtY3Cxc7WHirRarcjVEBERNQ+Di50L9nWDVCKgvFKN64XlYpdDRETULAwudk4mlaCFn24FXc5zISIi28bg4gBa1cxzSeedRUREZOMYXByAbul/rqBLRES2jsHFAegm6PKZRUREZOsYXByALrhcyytDpUotcjVERERNx+DiALw95HB3kUGj1eJKLifoEhGR7WJwcQCCIKBVzTyX9GscLiIiItvF4OIgQvjMIiIisgMMLg6CPS5ERGQPGFwcREgA13IhIiLbx+DiIEL83SEAKCyuRGFxhdjlEBERNQmDi4NwkcsQ4O0KAPjnqlLkaoiIiJrGqoLLpk2bMHr0aLHLsFu6FXQZXIiIyFZZTXDZsWMH1qxZI3YZdi20Zp7LJQYXIiKyUTKxC7h27RqWLl2Ko0ePok2bNmKXY9d0K+imMbgQEZGNEr3H5fTp03BycsInn3yCmJgYscuxa7pboi9fVSJPWS5yNURERI0neo9LcnIykpOTTXpOmUz0PGaVWvi7I8jXDdfySrFsy3E8PTIGbVsoxC7LoUmlEoPvJC62h/Vhm1gXa2gP0YOLqUkkAnx83MUuw2qtmNobz23+GZezivDCf09izqhu6NW5hdhlOTyFwlXsEqgWtof1YZtYFzHbw+6Ci0ajhVJZKnYZVstVJsHLM/pgxdaj+P3Cdbyw9RgeHtAeg3uGQRAEsctzOFKpBAqFK5TKMqjVGrHLcXhsD+vDNrEu5moPhcLV6F4cuwsuAKBS8Q/3rSgUTpg1Mgbbv/gL36dm4v0Df+NqbglGDewAqYTdsWJQqzX8c2tF2B7Wh21iXcRsD/4t5aCkEglGD+yAkcnhEAB8/+sVrPnoN5SWq8QujYiIqEEMLg5MEAQMig/D9BFdIHeS4HRaHla+exK5BWVil0ZERFQvBhdCXIcAzB8VBy8POTJzS7D8nRO4cKVQ7LKIiIjqELRarVbsIkxJrdYgL69E7DKslkwmgY+PO/LzS+qMT+Ypy7F2129Izy6Gk0yCJ4Z2QveOgSJV6hhu1R5keWwP68M2sS7mag9fX3ejJ+eyx4X0fBUumD8qDtHt/FCl0mDD3j+w/8g/sLNsS0RENozBhQy4Ossw8/5oDOgWCgDY/cNFbP3iT6h4GyIREVkBBheqQyIR8OjdHTDq7g4QBODH367itQ9PoaS8SuzSiIjIwTG4UIPu6haKmfdHw1kuxdlL+Vjxzklk53NxPyIiEg+DC91STLg/FoyKg4+nM7LySrH8nZP4O6NA7LKIiMhBMbjQbYUFeWLx2O5oHeyJ4rIqvPJ+Kn4+nSV2WURE5IAYXMgo3h7OmP9oHGLb+0Ol1uLNT8/gkx/TeMcRERFZFIMLGc1ZLsX0EV0wOD4MALD3xzS8/dkZVHFtBSIishAGF2oUiSDgoeRwjBkcAYkg4Mjpa1j9QSqKy3jHERERmR+DCzVJUtcQPPVQNFydpTiXUYjl75xAVh7vOCIiIvNicKEm69zWD/95rBv8FC7Izi/DindO4K/L+WKXRUREdozBhZolJMADi8Z2xx0tFSgpV2HVB7/i8O9XxS6LiIjsFIMLNZuXuxxzH4lF946BUGu02Lz/LPYcvAAN7zgiIiITY3Ahk5A7STHlvigMSWgNAPjsp0t485PTqFKpRa6MiIjsCYMLmYxEEHB/v3YYf09HSCUCjp3Nxsvvp0JZUil2aUREZCcYXMjk+kS3xKyRXeHmLMOFTCWWv3MCmbklYpdFRER2gMGFzCKytQ8WjumGQG9X5BaW44X/nsTpf/LELouIiGwcgwuZTQs/dywc0w3hoV4oq1DhtZ2n8MOvmWKXRURENozBhczK002OZx6ORa+oIGi0Wmz/8i98+N153nFERERNwuBCZuckk+CJoZ1wX2JbAMCXRy9j48d/oKKKdxwREVHjMLiQRQiCgPsS2+KJYZ0gkwo4eS4HL+34BQXFFWKXRkRENoTBhSwqISoYcx6OhYerE/7JKsLyd04gI7tY7LKIiMhGMLiQxXVo5Y2FY7ohyNcNecoKvPDuSfx24brYZRERkQ1gcCFRBPm4YeHobugY5o3ySjXW7jqFb05miF0WERFZOQYXEo2HqxNmjeyK3l2CodUCO74+h/cOnINGwzuOiIiofgwuJCqZVILH74nE/f3uAAAcOJGBlD2/o7xSJXJlRERkjRhcSHSCIGBIQhtMuS8KMqkEv57PxYvv/oI8ZbnYpRERkZVhcCGrER8ZhHmPxsLTzQmXs4ux/J0TuJRVJHZZRERkRRhcyKq0C/HCojHd0dLfHQXFlVi54yRS/84RuywiIrISDC5kdQK8XfGfx+IQ1cYHlVUapOz+Hf87dhlaPiaAiMjhMbiQVXJzccK/H4xBv64toQXwwbfn8e7/zkGt0YhdGhERiYjBhayWTCrBmEEReKh/OAQA36VmYu2u31BWwTuOiIgcFYMLWTVBEDC4Zximj+gCuUyCPy7m4YV3TyK3sEzs0oiISAQMLmQT4joEYN6oOHi5y5GZU4Ll75zEX5fzUaXi0BERkSORiV0AkbHatlBg8djuWPPRb8jIKcZL76UCABTucvh6OsPH0xm+CpfqnxXO8PWs/tnb0xkyKTM6EZE9YHAhm+KrcMGCx+Kw/cs/8cu5XKjUGihLKqEsqcQ/Daz5IgBQeFSHG19Plxuhpua7j6czvD3lkEoYboiIrB2DC9kcV2cZptzXGVqtFkVlVchXViCvqBz5RRXI0/1ca5tKrUVhcSUKiyuRdrWBcCMAXu7yGz02umCjqA42vp7O8PZwhkQiWPjTEhFRbQwuZLMEQYDCTQ6Fmxytgz3r3Uej1aK4tKpWmKlAnlIXcsqRV1SB/KIKqDVaFBRXoqC4EhcbuJ5EEODlITfoqbl5aMrLXc5wQ0RkRgwuZNckggCFuxwKdznaBNe/j0arRVFJZU2oqd17UxNslBUoKK4ON/k1QecClPWeSyoR4O0hh09NoLkxNHWj90bhLodEYLghImoKBhdyeNU9Kc7w8nBG2xb176PRaFFYUqkPNPlF1QEnT1mh/7mgqBJqjRbXlRW4rqxo8HpSiQCfmsnEfl4uCPb3gESrhdxJCle5FK7OMrg4S+Eql8HFWabf5iyXMvAQkcNjcCEygqRW2LijpaLefXTh5uahqOpem+qfdT03uYXlyC0sx98ZhQCuGV2Hiy7YyKVwkcvgqg84tYLOTdt0+7s63/iZd1kRka1icCEykdrhpiFqjQaFxZX6uTaFJZWoUGmRryxDabkK5RUqlFWqa76rUFahRlmFCmpN9XOayivVKK9UN7tWJ5kErnIpXHShRhdwbhV6avav/bNcJoHAXiAisiDRg4tGo0FKSgo++ugjFBUVoUePHliyZAlatWoldmlEJieVSKon9CpcgBAvyGQS+Pi4Iz+/BKpbLKZXpdKgrLIm2FSoUa4LNZU3wk5ZhQrlNdvKKlQor9lWOwhVVmn056tSaaAsrWrW55EIAlydpXCRS+Esl0EiCJBKBEgkAiQSQCrofhYgqflZWuvnG9tRc5wEEgH67dJa+0glAoSbjxdgsI/unPVdQ1qzv35bnXMJkMulqNAAxUXlgFYLqVRSfay0el+pRGBQIxKZ6MFlw4YNeO+99/Diiy8iODgYr7zyCiZOnIhPP/0Ucrlc7PKIrIKTTAInWfUdVM2h1mhQUamuFXrqCToNhJ4b+1Zv06J6YnNJuQol5SoADc/rsSe6ICORCJBJboQrqURiEHB0rw33k1S/d9N+klrbZLrXBvtJblzn5msY7Gu4n+5cuhr055UIdUKZLrwRWTtRg0tlZSW2bNmCOXPmICkpCQDw2muvoU+fPvjf//6HoUOHilkekd2RSiRwc5HAzcWpWefRarWoqFIb9P5UVKmh0Wih1mih0Wqh0dR81fysvum1RguDbWqNFtqbj9fWPq66h1ajqT5Oq9XWOd7guhot1FotNBrcdN26x+muq9UCaq0WarUGarUW2no+u0arhUZV/Y69RTWhpgdLWjs81YSg6pAj0fdSGQaom4KZ7hiJAKk+6Elu2vfmnyWGx9V8yZ2k8PBwQVlpBTQaQCJUL4Ug1PSeCTWvJaj5XmubIFQHTYlQd1vt78JN+0gEQJDU2oYb16t9fd1xZFmiBpc///wTJSUlSEhI0G9TKBTo1KkTjh8/zuBCZKUEQYCLXAYXuQxAw3N6bM3NQ3carRZqtRZqjQYajRYqTfXr6iCmgVqje1/3dYv9NDfOdfNrjab2Oeq/Zv37afQhTH9N7Y3tui/NTfU1FMq0WkCl1kKlbv48KkdSNwzdFIJqfUfNPsCNEFb9c/V7un1126rfrtlm8P5N22oKMTxn9TbB4Jq6czZ8TdSEtfquKZMKGNCrDUJ9Xc37S70FUYNLVlYWAKBFC8N7UAMDA/XvNYVMxjsmGiKtuZtEyrtKrALbw7rU2x7N65yyWrpQVh16bgSaG0HopqCkvin83BSg6mzTaGqFt7rh7ubAp99XW/daEqkEVTU9elpU9/hpNDXftdU9Zbrv1dt0+9zYphvaNNhWc1ztY7Va6K9jLF1PXc0r0zeWlbmaV4ZFY7uLdn1Rg0tZWRkA1JnL4uzsjMLCwiadUyIR4OPj3uza7J1CIV5aprrYHtaF7UHam4KQRls9pFg7KNUOQZp6wpRuOLN2mKodkLS4cS7cFL7072tqtqOe42uft56aofuOG7Xpg1/t929zntqvASA+KljU/0ZEDS4uLi4Aque66H4GgIqKCri6Nu2XotFooVSWmqQ+eySVSqBQuEKpLINa3fBdLGQZbA/rwvawPrbSJpKaL9QMr1QzeGEXzNUeCoWr0T3PogYX3RBRdnY2wsLC9Nuzs7MRERHR5PPe6rZSqqZWa/h7siJsD+vC9rA+bBPrImZ7iDqw3rFjR3h4eODo0aP6bUqlEmfOnEGPHj1ErIyIiIiskag9LnK5HI899hhWrVoFX19fhISE4JVXXkFwcDAGDhwoZmlERERkhURfgG7mzJlQqVRYtGgRysvL0aNHD2zevBlOTnY6lZ+IiIiaTNDqpgnbCbVag7y8ErHLsFrGLjFPlsH2sC5sD+vDNrEu5moPX193oyfncvEIIiIishkMLkRERGQzGFyIiIjIZjC4EBERkc1gcCEiIiKbweBCRERENoPBhYiIiGwGgwsRERHZDLtbgE5b86hxaphUKrHqp6w6GraHdWF7WB+2iXUxR3tIJAIEwbgnadtdcCEiIiL7xaEiIiIishkMLkRERGQzGFyIiIjIZjC4EBERkc1gcCEiIiKbweBCRERENoPBhYiIiGwGgwsRERHZDAYXIiIishkMLkRERGQzGFyIiIjIZjC4EBERkc1gcCEiIiKbweDiIAoKCrBkyRL07dsXcXFxeOSRR3DixAmxyyIAaWlpiI2NxZ49e8QuxeHt3bsX99xzD7p06YIhQ4bgiy++ELskh6VSqbB27Vr0798fsbGxGDVqFH799Vexy3JImzZtwujRow22nT17Fo899hi6du2K5ORkvPPOOxarh8HFQcyaNQupqal49dVXsXv3bkRGRmLChAm4ePGi2KU5tKqqKsyZMwelpaVil+Lw9u3bh4ULF2LUqFHYv38/hg4dqv/vhixv48aN+Oijj/D8889j7969aNu2LSZOnIjs7GyxS3MoO3bswJo1awy25efnY/z48QgLC8Pu3bsxffp0rFq1Crt377ZITQwuDuDSpUs4fPgwli1bhu7du6Nt27ZYvHgxAgMD8emnn4pdnkNbt24dPDw8xC7D4Wm1WqxduxZjxozBqFGjEBYWhqlTp+LOO+/EsWPHxC7PIR04cABDhw5FYmIiWrdujfnz56OoqIi9LhZy7do1TJkyBatWrUKbNm0M3vvwww/h5OSE5557Du3atcP999+PcePG4c0337RIbQwuDsDHxwdvvvkmunTpot8mCAIEQYBSqRSxMsd2/Phx7Ny5Ey+++KLYpTi8tLQ0ZGZmYtiwYQbbN2/ejMmTJ4tUlWPz8/PDd999h4yMDKjVauzcuRNyuRwdO3YUuzSHcPr0aTg5OeGTTz5BTEyMwXsnTpxAfHw8ZDKZfluvXr3wzz//IDc31+y1Mbg4AIVCgX79+kEul+u3ffXVV7h06RL69OkjYmWOS6lUYu7cuVi0aBFatGghdjkOLy0tDQBQWlqKCRMmICEhAQ8++CC+/fZbkStzXAsXLoSTkxPuuusudOnSBa+99hpef/11hIWFiV2aQ0hOTsa6devQqlWrOu9lZWUhODjYYFtgYCAA4OrVq2avjcHFAf3yyy9YsGABBg4ciKSkJLHLcUjLli1DbGxsnX/hkziKi4sBAPPmzcPQoUOxZcsW9O7dG9OmTcORI0dErs4xnT9/Hp6enli/fj127tyJESNGYM6cOTh79qzYpTm88vJyg38IA4CzszMAoKKiwuzXl91+F7InBw4cwJw5cxAXF4dVq1aJXY5D2rt3L06cOMH5RVbEyckJADBhwgQMHz4cABAZGYkzZ85g69atSEhIELM8h3P16lXMnj0b27ZtQ/fu3QEAXbp0wfnz57Fu3Tps2LBB5Aodm4uLCyorKw226QKLm5ub2a/PHhcH8u677+LJJ59E//798cYbb+gTMlnW7t27cf36dSQlJSE2NhaxsbEAgKVLl2LixIkiV+eYgoKCAAAdOnQw2B4eHo6MjAwxSnJop06dQlVVlcG8PACIiYnBpUuXRKqKdIKDg+vc3aV7rftvyZzY4+Ig3nvvPTz//PMYPXo0Fi5cCEEQxC7JYa1atQrl5eUG2wYOHIiZM2fi3nvvFakqxxYVFQV3d3ecOnVK/y98ADh37hznVIhAN3/ir7/+QnR0tH77uXPn6tzhQpbXo0cPfPDBB1Cr1ZBKpQCAn3/+GW3btoWfn5/Zr8/g4gDS0tLwwgsv4O6778bkyZMNZn27uLjA09NTxOocT0P/IvHz87PIv1aoLhcXF0ycOBHr169HUFAQoqOjsX//fhw+fBjbtm0TuzyHEx0djW7dumHevHlYunQpgoODsXfvXhw5cgTvv/++2OU5vPvvvx9vv/02Fi5ciIkTJ+K3337Dtm3b8Oyzz1rk+gwuDuCrr75CVVUVvv76a3z99dcG7w0fPpy34xIBmDZtGlxdXfHaa6/h2rVraNeuHdatW4eePXuKXZrDkUgk2LhxI9asWYMFCxagsLAQHTp0wLZt2+rcmkuW5+fnh7fffhsrVqzA8OHDERAQgLlz5+rnh5mboNVqtRa5EhEREVEzcXIuERER2QwGFyIiIrIZDC5ERERkMxhciIiIyGYwuBAREZHNYHAhIiIim8HgQkRERDaDwYWIyAK4ZBaRaTC4EFmh0aNHo1OnTvj999/rfT85ORnz58+3SC3z589HcnKyRa7VGCqVCvPnz0dsbCzi4uLw888/N7hvRUUFtm3bhvvvvx/dunVDfHw8Hn74Yezdu9cgUKxbtw4REREmrbOyshIvvPACnwZOZCIMLkRWSq1WY8GCBXUeH0/VDh06hI8//hjjxo3Dpk2b6jxJWCc3NxcjR47Exo0b0b9/f7z22mt4+eWXERERgfnz52Px4sVm7Q3Jzs7G9u3boVKpzHYNIkfCZxURWSlPT0/8/fffWL9+PZ5++mmxy7E6BQUFAIARI0agVatWDe43b948ZGVlYefOnQZPFk5KSkLLli3x6quvon///rjrrrvMXDERmQJ7XIisVGRkJP7v//4Pb7/9Nv74449b7hsREYF169YZbLt52GP+/PmYMGECdu7ciQEDBiA6OhoPP/ww0tLS8N1332HYsGGIiYnBgw8+iLNnz9a5xs6dO5GUlITo6GiMHTsWZ86cMXj/ypUrmDVrFuLj4xETE1Nnn4yMDERERGDr1q0YPHgwYmJisHv37no/j1qtxo4dOzBs2DBER0cjKSkJq1atQkVFhf6z6IbKBgwYgNGjR9d7nrNnz+LHH3/EhAkTDEKLzrhx4zBq1Ci4ubnVe3x9Q3J79uxBREQEMjIyAADl5eVYtmwZ+vbti86dO2Pw4MHYvHmz/jPrAtGCBQsMhtxOnDiBxx57DDExMYiPj8e8efOQl5dncJ1OnTrho48+Qu/evREfH4/z58/j8uXLmDJlCnr27ImYmBiMHDkSP/zwQ731E9kj9rgQWbH//Oc/OHz4MBYsWIDdu3dDLpc363ypqanIzs7G/PnzUVFRgWXLlmHSpEkQBAEzZ86Eq6srli5dijlz5mD//v3647KyspCSkoLZs2fDw8MDKSkpGD16ND799FO0bNkSeXl5ePjhh+Hq6orFixfD1dUV27dvx6hRo7Br1y60a9dOf65169Zh4cKF8PDwaPBJv0uWLMG+ffvwxBNPoHv37jhz5gzWr1+Ps2fP4u2338a0adMQHByMjRs3IiUlBW3btq33PIcOHQKABufoODs7Y8mSJU39dQIAXnjhBfz444+YN28e/P39cfDgQbz88svw9vbGsGHDkJKSghkzZmDq1KkYOHAgAOD48eMYP348evXqhTVr1qCwsBBr167FmDFjsGvXLri4uACoDnBbtmzBihUrkJ+fj7Zt22Lo0KEIDAzEyy+/DJlMhnfeeQdTp07FF198gdatWzfrsxDZAgYXIivm5eWF5557DlOnTjXJkFFJSQnWrFmjDxLHjh3DBx98gG3btiEhIQEAcOnSJbz00ktQKpVQKBQAqv8CXb9+PaKjowEAMTExGDBgAP773/9i3rx52L59OwoKCvD+++8jJCQEANC3b1/cc889WLt2LV5//XV9Df/6179w//33N1jj+fPnsWvXLsyePRuTJk0CAPTu3RuBgYGYO3cuDh48iH79+iEsLAxAdc9UaGhovee6evUqADT4vikcO3YMvXv3xpAhQwAAPXv2hJubG/z8/CCXyxEZGQkACAsLQ6dOnQAAq1evRtu2bbFp0yZIpVIA1b/TIUOGYPfu3Rg1apT+/FOmTEFSUhIAICcnBxcvXsS0adPQr18/AEB0dDRSUlI4F4ocBoeKiKxccnIy7r33Xrz99ts4ffp0s87l5eVl0Pvh7+8PAAY9H97e3gAApVKp39aqVSt9aAGAgIAAdO3aFcePHwcAHDlyBJGRkQgKCoJKpYJKpYJEIkHfvn3x008/GdSg+4u8IceOHQMAfRDQGTJkCKRSKY4ePWrsx9WHArVabfQxjdWzZ098+OGHeOKJJ/Duu+8iPT0d06dP14eNm5WVleHUqVPo168ftFqt/vfVqlUrtGvXDocPHzbYv/bvy9/fH+Hh4Vi8eDHmzZuHTz/9FBqNBgsWLED79u3N9hmJrAl7XIhswKJFi3DkyBH9kFFTeXh41Lu9oTkeOrqAU5ufn5++R6OgoACXLl1CVFRUvceXlZUZfa3CwkIA1eGoNplMBh8fHxQVFd3y+Np0vT9XrlxBeHh4vftcu3YNgYGBEATB6PPWtnDhQgQHB+OTTz7B888/j+effx6xsbFYtmwZOnbsWGd/pVIJjUaDt956C2+99Vad952dnQ1e1/59CYKALVu2YOPGjfj666+xd+9eODk5YcCAAXj22Wfh5eXVpM9AZEsYXIhsgJeXF5YtW4bp06djw4YN9e5zc69CaWmpya6vCxO15eTkwNfXF0D1HVDx8fGYO3duvcc3Zm6O7i/fnJwcffAAgKqqKuTn58PHx8focyUmJgIAfvjhh3qDi0qlwn333Ye4uLgm/17lcjmmTp2KqVOn4sqVK/juu++wYcMGzJ4922CekI67uzsEQcC4cePq9CoBgKur6y0/U1BQEJYtW4alS5fizz//xJdffom33noLPj4+WLp06S2PJbIHHCoishEDBgzA0KFD8eabbxrcfQJU96Rcu3bNYNsvv/xismunpaXh8uXL+tdXr15FamoqevbsCQCIj49HWloa2rZtiy5duui/9u3bh127dumHbIwRHx8PAHX+0t+/fz/UajW6detm9Lnat2+Pvn374q233kJ6enqd9zdt2oT8/Hzce++99R7v4eGBrKwsg20nT57U/1xeXo5BgwZhy5YtAICWLVti1KhRGDJkCK5cuQIAdT67h4cHOnXqhIsXLxr8rtq3b49169bdcigsNTUVd955J3777TcIgoDIyEg8/fTT6NChg/56RPaOPS5ENmTx4sX4+eefkZuba7A9KSkJ+/fvR0xMDFq3bo09e/bg0qVLJruus7Mzpk6diqeffhpqtRpr166Ft7c3xo4dC6D6tuJ9+/Zh3LhxePzxx+Hj44PPP/8cH374IRYsWNCoa4WHh2P48OF4/fXXUVZWhh49euDs2bNISUlBz5490adPn0ad79lnn8XYsWPx0EMPYcyYMYiJiUFJSQm+/PJL7N+/Hw8//DAGDx5c77H9+/fHpk2bsGnTJsTExODbb781WKHXxcUFUVFRSElJgZOTEyIiIpCWloaPP/4YgwYNAlDdGwVUzwNq164dYmJiMGvWLEyaNAmzZ8/Gvffeq7976NSpU5g2bVqDn6VTp05wcXHB3Llz8eSTT8Lf3x8//fQTzp49izFjxjTq90JkqxhciGyIt7c3li1bhhkzZhhsX7BgAVQqFV566SXIZDLcc889mD17NhYtWmSS63bq1AmDBg3CsmXLUFRUhISEBPznP//RDxUFBQXhgw8+wOrVq7Fs2TJUVFSgTZs2WLFiBR544IFGX2/FihVo3bo1du/ejbfeeguBgYEYM2YMpk2bBomkcR3FLVu2xM6dO7F9+3Z89tlnePPNNyGXy3HHHXdg9erVuOeeexo8dvLkycjLy8PmzZtRVVWFpKQkrFixAlOnTtXv89xzz2HNmjXYsmULcnJy4OfnhwceeAD//ve/AVT3sIwfPx47d+7EDz/8gMOHDyMxMRGbN29GSkoKZs6cCScnJ0RFRWHr1q3o2rVrg/U4Oztjy5YtWL16NVasWAGlUok2bdrgueeew4gRIxr1eyGyVYKWT/4iIiIiG8E5LkRERGQzGFyIiIjIZjC4EBERkc1gcCEiIiKbweBCRERENoPBhYiIiGwGgwsRERHZDAYXIiIishkMLkRERGQzGFyIiIjIZjC4EBERkc34f+FFeLzFQtD2AAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plot an elbow graph\n",
    "\n",
    "sns.set()\n",
    "plt.plot(range(1, 11), wcss)\n",
    "plt.title('The Elbow Point Graph')\n",
    "plt.xlabel('Number of Clusters')\n",
    "plt.ylabel('WCSS')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ifedd9_gQC4x"
   },
   "source": [
    "Optimum Number of Clusters = 5"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "J3DiuWPtQKnU"
   },
   "source": [
    "Training the k-Means Clustering Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "-5S3HwLpPy3h",
    "outputId": "4d008806-2579-4c85-8b65-d53f10f13191"
   },
   "outputs": [],
   "source": [
    "kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)\n",
    "Y = kmeans.fit_predict(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ehXD5VrDSyuB"
   },
   "source": [
    "5 Clusters -  0, 1, 2, 3, 4"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "AfFa2VDQRNcK"
   },
   "source": [
    "Visualizing all the Clusters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 518
    },
    "id": "Tl_Obm0aQ_cU",
    "outputId": "9e554efe-7307-4f24-bbda-7fded2ce0616"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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xIgReOutt5CXl4f169dDo9Fg7NixAPLr0d3d3fHpp5/i/fffR1ZWFlatWmUcwS7o3LlzmDp1Kl5++WX8/vvvWL16NV5//XXUqFED1apVg7u7Oz744AOMHz8eAQEBOHHiBK5evWq2nURUOTFQExHZ0OjRo9GoUSNs27YNCxcuRFpaGoKDg9GpUyfjVHQGH330EbRaLRYtWgSFQoFevXphypQpJgucREZGYtmyZVi5ciVSUlIQHByMcePGYcSIEQCAunXrok+fPti2bRt++eUX7N+/3xjytm7dip07d6JKlSro0aMHJk6caLau2ZwGDRpg27ZtWL58OT744AOIooh69erh888/R5cuXUp1zkGDBqFNmzb45ptvsGnTJiQmJkIul6NWrVoYPHgwBg4cWKim+XHt2rXDpk2bsGrVKkyePBmurq5o1aoVFi1aZLxR0NvbG6tXr8bSpUsxduxYPPnkkxg3bpxxXu+C3n33Xdy9exfjxo2Dr68vRo0aZXwT4+bmho0bNxpnRElPT0doaCjmzZuHV199tVSvARFVbIJY1GdsRERETqp+/foYN24cxo8f7+imEFEFwRpqIiIiIiIJGKiJiIiIiCRgyQcRERERkQQcoSYiIiIikoCBmoiIiIhIAgZqIiIiIiIJGKiJiIiIiCTgwi4OJIoi9HreE1oWZDKBr7WTYt87L/a9c2K/Oy979L1MJkAQBLPHMVA7kF4vIjk5y9HNqPQUChl8fT2Rnp4NrVbv6OZQGWLfOy/2vXNivzsve/W9n58n5HLzgZolH0REREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEnCWDyIiIiIL6PV66HRaRzeDiqDXC8jJkUOjyYVOZ9nUeXK5AjKZbcaWGaiJiIiISiCKItLTk6FWZzq6KVSC+/dl0OutmzJPqVTB29vPormmS8JATURERFQCQ5hWqXzh6uomOXyRfcjlgsWj06IoQqPJRWZmCgDAx8df0rUZqImIiIiKodfrjGFapfJ2dHOoBAqFzKpFXVxd3QAAmZkp8PLylVT+wZsSiYiIiIqh0+kAPApfVLkY+lVqbTwDNREREZEZLPOonGzVrwzUREREREQSMFATEREROQmtVotdu75BePjbePHFjujTpysmTRqLM2dOmRz33HOtcPDgv2123QsXzuH8+XM2O9/j9Ho9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TLbt2rULLi4umDdvHhQKBWrXro2bN29i/fr16N+/PzQaDTZu3IipU6eiU6dOAIDly5ejQ4cOOHr0KPr06VMm5yAiIiIi51UuRqhPnjyJnTt34tNPPzXZfurUKbRp0wYKxaPc37ZtW8THx+P+/fv4/fffkZWVhXbt2hn3e3t7o1GjRjh58mSZnYOIiIiInJfDR6jT09PxwQcfYMaMGQgODjbZl5iYiHr16plsq1atGgAgISEBiYmJAFDocdWqVTPuK4tzBAQEWPGMTSkU5eI9TaUml8tM/k/Og33vvNj3zon97rwc3fcOD9Rz5sxBixYt8NJLLxXal5OTA1dXV5Ntbm5uAIDc3Fyo1WoAKPKYtLS0MjtHaclkAnx9PUv9eLKOt7fS0U0gB2HfOy/2vXNivzsvR/W9QwP13r17cerUKfz73/8ucr+7u7vxxkADQ4D18PCAu7s7AECj0Rj/bThGqVSW2TlKS68XkZ6ebf5AkkQul8HbW4n0dDV0Or2jm0NliH3vvNj3zon97rzs1ffe3kqLRr0dGqijo6Px4MED482ABrNnz8bBgwcRFBSEe/fumewzfB0YGAitVmvcFhISYnJM/fr1AaBMziGFVssf+LKi0+n5ejsp9r3zYt87J/a783JU3zs0UC9ZsgQ5OTkm27p164YJEyagb9+++P7777Fjxw7odDrI5XIAQGxsLGrVqgV/f394eXlBpVIhLi7OGIbT09Nx5coVDB48GADQunVru5+DiIiIiJyXQ6v2AwMDUbNmTZP/AMDf3x+BgYHo378/MjMz8cknn+Cvv/7C7t27sXnzZowcORJAft3z4MGDsWTJEhw/fhy///47Jk2ahKCgIHTr1g0AyuQcREREROS8HH5TYkn8/f2xYcMGREREoF+/fqhatSo++OAD9OvXz3jMhAkToNVqMWPGDOTk5KB169aIioqCi4tLmZ6DiIiIiJyTIIqi6OhGOCudTo/k5CxHN6PSUyhk8PX1REpKFmvqnAz73nmx750T+9152avv/fw8LbopkRM1EhERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJoCjNg5KTkxEVFYUTJ04gKSkJGzZswLFjx9CgQQN07drV1m0kIiIiIiq3rB6hvnXrFvr27Ytdu3YhMDAQDx48gE6nw40bNzBhwgT89NNPdmgmEREREVH5ZPUI9aJFi+Dv74+vv/4aHh4eaNy4MQBg6dKlyM3Nxdq1a9GpUydbt5OIiIiIqFyyeoQ6JiYGY8aMgbe3NwRBMNk3cOBA/PnnnzZrHBERERFReVeqmxIViqIHtjUaTaGQTURERERUmVkdqFu1aoV169YhOzvbuE0QBOj1enzzzTdo2bKlTRtIRERERFSeWV1DPWXKFLz55pvo1q0bwsLCIAgCoqKicO3aNdy8eRPbt2+3RzuJiIiIiMolq0eo69Wrh+joaISFhSEuLg5yuRwnTpxASEgIduzYgYYNG9qjnURERERE5ZLVI9R79uxB+/btsXTpUnu0h4iIiIioQrF6hHrevHm4cOGCPdpCRERERFThWB2og4KCkJmZaY+2EBERERFVOFaXfAwcOBARERE4e/Ys6tevD09Pz0LHvPLKK7ZoGxERERFRuWd1oP70008BALt27SpyvyAIDNRERERE5DSsDtTHjx+3RzuIiIiIiCokqwP1k08+afy3Wq1GZmYmqlSpAhcXF5s2jIiIiIioIrA6UAPAqVOn8Nlnn+HSpUsQRREA0LRpU0yaNAlt27a1aQOJiIiIiMozqwP1mTNnMGTIENSoUQNjxoxBQEAA7t27hwMH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FwQBCxcutFkDiYiIiIjKM6sD9Z49e+Dl5QW9Xl/kTB+CINikYUREREREFYHVgfrHH3+0RzuIiIiIiCokznlHRERERCSBRSPUXbp0weeff44GDRqgc+fOJZZ1CIKAY8eO2ayBRERERETlmUWBuk2bNvD09DT+m3XSRERERET5BJHz3zmMTqdHcnKWo5tR6SkUMvj6eiIlJQtard7RzaEyxL53Xux758R+d1726ns/P0/I5eYrpFlDTUREREQkgUUlHw0aNLCqzOPq1aulbhARERERUUViUaAeO3asMVDn5uZi06ZNCA0NRffu3VG1alWkpqbixx9/xP/+9z+MHj3arg0mIiIiIipPLArU48ePN/77448/RqdOnbB69WqTUetRo0Zh2rRpuHz5su1bSURERERUTlldQ33o0CEMHDiwyBKQl19+Gb/88otNGkZEREREVBFYHag9PT3x999/F7nvypUr8PHxkdwoIiIiIqKKwuqlx3v37o1ly5bBxcUFnTp1gq+vLx48eIDDhw/j888/x/Dhw+3RTiIiIiKicsnqQD1lyhQkJCRg1qxZJmUfoiji9ddfx9ixY23aQCIiIiKi8szqQO3q6opVq1bhzz//xKlTp5Ceng5fX1+0bdsWISEh9mgjEREREVG5ZXWgNqhbty7q1q1ry7YQEREREVU4VgdqURTx7bff4j//+Q/UajX0etPlHQVBwFdffWWzBhIRERERlWdWB+qlS5diw4YNqF69OoKCggpNnyeKos0aR0RERERU3lkdqPfu3YuhQ4di+vTp9mgPEREREVGFYvU81JmZmejUqZMdmkJEREREVPFYHaifeeYZnDlzxh5tISIiIiKqcKwu+Rg2bBimTZsGrVaLZs2aQalUFjqmdevWNmkcEREREVF5Z3WgHjp0KADg888/B4BCi7sIgoCrV6/aqHlEREREROWb1YF6y5Yt9mgHEREREVGFZHWgbtOmjT3aQURERERUIZVqpcTk5GRERUXhxIkTSEpKwoYNG3Ds2DE0aNAAXbt2tXUbiYiIiIjKLatn+bh16xb69u2LXbt2ITAwEA8ePIBOp8ONGzcwYcIE/PTTT3ZoJhERERFR+WT1CPWiRYvg7++Pr7/+Gh4eHmjcuDGA/BUUc3NzsXbtWs5TTUREREROw+oR6piYGIwZMwbe3t6Flh0fOHAg/vzzT5s1joiIiIiovLM6UAOAQlH0wLZGoykUsomIiIiIKjOrA3WrVq2wbt06ZGdnG7cJggC9Xo9vvvkGLVu2tGkDiYiIiIjKM6trqKdMmYI333wT3bp1Q1hYGARBQFRUFK5du4abN29i+/bt9mgnEREREVG5ZPUIdb169RAdHY2wsDDExcVBLpfjxIkTCAkJwY4dO9CwYUN7tJOIiIiIqFwq1TzUoaGhWLp0qa3bQkRERERU4ZQqUOfm5mLv3r347bffkJaWBn9/f7Rr1w59+vQp9oZFIiIiIqLKyOr0e+fOHbzzzju4ffs2atSoAX9/f8THx+P777/H5s2b8dVXX8HHx8cebSUiIiIiKnesDtQRERHQ6/XYs2ePSb30pUuXMH78eHz22WeIiIiwaSOJiIiIiMorq29KjIuLw9SpUwvdfNi4cWNMnDgRx48ft1njiIiIiIjKO6sDtZubG+RyeZH7VCoVRFGU3CgiIiIioorC6kD9zjvvYNmyZfjnn39MtqelpWHt2rV45513bNY4IiIiIqLyzuoa6vj4eKSkpKBHjx545plnEBgYiJSUFJw+fRpqtRru7u6Ii4sDkL+C4ldffWXzRhMRERERlRdWB+rbt2+jfv36AACdToc7d+4AABo1amQ8xlD2wfIPIiIiIqrsrA7UX3/9tT3aQURERERUIVldQ/24tLQ0XLx4ERkZGbZoDxERERFRhWJxoL5w4QJGjRqFvXv3Grd9/fXX6NixI15//XV06NABUVFR9mgjEREREVG5ZVGg/v333/H222/j6tWr8PDwAABcvHgRCxcuRI0aNbB69WqMGTMGy5cvx7Fjx+zaYCIiIiKi8sSiGup169ahQYMG2Lx5M5RKJQBgy5YtAIAlS5agQYMGAID79+/j66+/RteuXe3UXCIiIiKi8sWiEeqTJ0/i7bffNoZpAPj1119Ro0YNY5gGgOeeew5XrlyxfSuJiIiIiMopiwJ1amoqgoKCjF9fu3YNKSkpCAsLMzlOqVRCo9HYtoVEREREROWYRYG6SpUqePDggfHr2NhYCIKAdu3amRx37do1+Pn52baFRERERETlmEWBuk2bNti1axdEUYRWq0V0dDTc3NzQoUMH4zEajQbbtm1Dy5Yt7dZYIiIiIqLyxqKbEkePHo2BAweia9euEEURd+7cwdixY+Hl5QUAiI6OxrZt23Djxg189tlndm0wEREREVF5YlGgrlu3Lnbt2oWNGzfiwYMHGD58ON58803j/hUrVkChUODzzz9Hw4YN7dZYIiIiIqLyxuKlx+vUqYOFCxcWue+7775D1apVIZNJXniRiIiIiKhCsThQlyQwMNAWpyEiIiIiqnA4pExEREREJAEDNRERERGRBAzUREREREQSMFATEREREUnAQE1EREREJAEDNRERERGRBAzUREREREQSMFATEREREUnAQE1EREREJAEDNRERERGRBAzUREREREQSMFATEREREUnAQE1EREREJAEDNRERERGRBAzUREREREQSODxQp6amYtasWejYsSNatmyJN998E6dOnTLuj4mJwauvvopmzZqhR48eOHDggMnjc3NzMXfuXLRr1w4tWrTAlClTkJycbHJMWZyDiIiIiJyTwwP15MmTcfbsWSxbtgzR0dFo2LAhwsPDcf36dVy7dg0jR45Ehw4dsHv3brz22mv44IMPEBMTY3z8nDlz8Ouvv2L16tX46quvcP36dUyYMMG4v6zOQURERETOSRBFUXTUxW/evIlu3bph+/bteOaZZwAAoiiiW7du6NOnDx48eICrV6/i22+/NT5mypQpSE1NRVRUFO7evYtOnTph7dq1eP755wEAN27cQI8ePbBjxw60aNECs2bNsvs5Skun0yM5OavUjyfLKBQy+Pp6IiUlC1qt3tHNoTLEvnde7HvnxH53Xvbqez8/T8jl5sefHTpC7evri/Xr16NJkybGbYIgQBAEpKen49SpU2jXrp3JY9q2bYvTp09DFEWcPn3auM2gVq1aCAwMxMmTJwGgTM5BRERERM5L4ciLe3t7G0eFDY4cOYKbN2/i448/xp49exAUFGSyv1q1alCr1UhJScHdu3fh6+sLNze3QsckJiYCABITE+1+Dj8/v1K/BgqFw6tuKj3DO0tL3mFS5cK+d17se+fEfndeju57hwbqx505cwYfffQRunXrhk6dOiEnJweurq4mxxi+1mg0UKvVhfYDgJubG3JzcwGgTM5RWjKZAF9fz1I/nqzj7a10dBPIQdj3zot975zY787LUX1fbgL1sWPHMHXqVLRs2RJLliwBkB9qHw+shq+VSiXc3d2LDLS5ublQKpVldo7S0utFpKdnl/rxZBm5XAZvbyXS09XQ6VhT50zY986Lfe+c2O/Oy1597+2ttGjUu1wE6q1btyIiIgI9evTAokWLjKO/wcHBuHfvnsmx9+7dg4eHB7y8vBAUFITU1FRoNBqTEeR79+4hMDCwzM4hBW+aKDs6nZ6vt5Ni3zsv9r1zYr87L0f1vcOLjLZv34758+dj0KBBWLZsmUmobdWqFX777TeT42NjY9GyZUvIZDI888wz0Ov1xhsLgfwZOu7evYvWrVuX2TmIiIiIyHk5NA3euHEDCxcuxIsvvoiRI0fi/v37SEpKQlJSEjIyMvD222/jwoULWLJkCa5du4aNGzfi8OHDGDZsGAAgMDAQvXv3xowZMxAXF4cLFy5g8uTJaNOmDZo3bw4AZXIOIiIiInJeDp2Heu3atVi+fHmR+/r164dPP/0UP//8MxYvXoz4+HhUr14d48ePR69evYzHZWdnY+HChThy5AgAoGPHjpgxYwZ8fX2Nx5TFOUqD81CXDc5L6rzY986Lfe+c2O/Oy9HzUDs0UDs7BuqywV+wzot977zY986J/e68HB2oWQBMRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCQBAzURERERkQQM1EREREREEjBQExERERFJwEBNRERERCRBuQrU69atw9tvv22y7erVqxg8eDCaN2+Ozp07Y8uWLSb79Xo9Vq1ahQ4dOqB58+YYPnw4bt26VebnICIiIiLnVG4C9bZt27BixQqTbSkpKRg6dChCQkIQHR2NsWPHYsmSJYiOjjYes2bNGmzfvh3z58/Hjh07oNfrMWzYMGg0mjI9BxERERE5J4WjG3D37l3Mnj0bcXFxCA0NNdm3a9cuuLi4YN68eVAoFKhduzZu3ryJ9evXo3///tBoNNi4cSOmTp2KTp06AQCWL1+ODh064OjRo+jTp0+ZnIOIiIiInJfDR6gvX74MFxcX7Nu3D82aNTPZd+rUKbRp0wYKxaPc37ZtW8THx+P+/fv4/fffkZWVhXbt2hn3e3t7o1GjRjh58mSZnYOIiIiInJfDR6g7d+6Mzp07F7kvMTER9erVM9lWrVo1AEBCQgISExMBAMHBwYWOMewri3MEBARY8EyJiIiIqDJyeKAuSU5ODlxdXU22ubm5AQByc3OhVqsBoMhj0tLSyuwcUigUDv+QoNKTy2Um/yfnwb53Xux758R+d16O7vtyHajd3d2NNwYaGAKsh4cH3N3dAQAajcb4b8MxSqWyzM5RWjKZAF9fz1I/nqzj7a10dBPIQdj3zot975zY787LUX1frgN1UFAQ7t27Z7LN8HVgYCC0Wq1xW0hIiMkx9evXL7NzlJZeLyI9PbvUjyfLyOUyeHsrkZ6uhk6nd3RzqAyx750X+945sd+dl7363ttbadGod7kO1K1bt8aOHTug0+kgl8sBALGxsahVqxb8/f3h5eUFlUqFuLg4YxhOT0/HlStXMHjw4DI7hxRaLX/gy4pOp+fr7aTY986Lfe+c2O/Oy1F9X66LjPr374/MzEx88skn+Ouvv7B7925s3rwZI0eOBJBf9zx48GAsWbIE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",
      "text/plain": [
       "<Figure size 800x800 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting all the clusters and their Centroids\n",
    "\n",
    "plt.figure(figsize=(8,8))\n",
    "plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Cluster 1')\n",
    "plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Cluster 2')\n",
    "plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Cluster 3')\n",
    "plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='violet', label='Cluster 4')\n",
    "plt.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='blue', label='Cluster 5')\n",
    "\n",
    "# Plot the centroids\n",
    "plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')\n",
    "\n",
    "plt.title('Customer Groups')\n",
    "plt.xlabel('Age')\n",
    "plt.ylabel('Spending Score')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "OjWc3GPiUFBm"
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
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
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
