# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 23:36:31 2021

@author: orkun
"""

from tkinter import filedialog
import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt


root = tk.Tk()
root.wm_attributes('-topmost', 1)
root.withdraw()


class file_read(object):
    def __init__(self, file_path, miss_format, delimiter):
        self.file_path = file_path
        self.miss_format = miss_format
        self.delimiter = delimiter


class miss_columns(file_read):
    def __init__(self, file_path, miss_format, delimiter):
        super().__init__(file_path, miss_format, delimiter)

    def missing_cols_get(self):
        df = pd.read_csv(self.file_path, delimiter=self.delimiter, na_values=self.miss_format)
        self.df = df
        self.missing_col_list = [column for column in df.columns if df[column].isnull().any()]
        return self.missing_col_list

    def imput_data_get(self):
        num_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        cat_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

        imputed_data = self.df
        for col_name in self.missing_col_list:
            if is_numeric_dtype(self.df[col_name].dtypes):
                imputer = num_imp.fit_transform(pd.DataFrame(self.df[col_name]))
                imputed_data[col_name] = pd.DataFrame(data=imputer)
            else:
                imputer = cat_imp.fit_transform(pd.DataFrame(self.df[col_name]))
                imputed_data[col_name] = pd.DataFrame(data=imputer)

        return imputed_data


class Lin_Regression(object):
    def __init__(self, dataframe):
        self.df = dataframe

    def lbl_encoding(self):
        from sklearn.preprocessing import LabelEncoder

        cat_columns = self.df.select_dtypes("object").columns
        labelencoder = LabelEncoder()

        for col in cat_columns:
            self.df[col] = labelencoder.fit_transform(self.df[col])

        return self.df

    def one_hot_encoding(self):

        cat_columns = self.df.select_dtypes("object").columns
        # generate binary values using get_dummies
        for col in cat_columns:
            self.df = pd.concat([pd.get_dummies(self.df[col], prefix='Type', drop_first=True), self.df], axis=1).drop([col], axis=1)
        return self.df

    def train_test_split(self):

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X = self.df.iloc[:, :-1].values
        y = self.df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def lin_reg_process(self):
        # Importing the Linear Regression libraries and Metrics
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.metrics import classification_report, confusion_matrix

        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)

        # Predicting the Test set results
        y_pred = regressor.predict(self.X_test)

        # The coefficients
        print('Coefficients: \n', regressor.coef_)

        # The mean squared error
        print('Mean squared error: %.2f' % mean_squared_error(self.y_test, y_pred))

        # The coefficient of determination: 1 is perfect prediction
        print('Coefficient of determination: %.2f' % r2_score(self.y_test, y_pred))

        # # Visualising the Linear Regression results
        plt.scatter(self.X_test, self.y_test, color='red')
        plt.plot(self.X_test, y_pred, color='blue')


if __name__ == '__main__':
    # ---Read File Path
    file_path = filedialog.askopenfilename(parent=root, initialdir="/", title='Please select a file')
    missing_formats = ["NA", "nan", "NAN", ""]
    delimeter = ","
    missing_object = miss_columns(file_path, missing_formats, delimeter)
    missing_columns = missing_object.missing_cols_get()
    print(file_path + " has missing columns : " + ",".join(missing_columns))
    imput_data = missing_object.imput_data_get()
    # print(imput_data)
    reg_obj = Lin_Regression(imput_data)
    one_enc_data = reg_obj.one_hot_encoding()
    # lbl_enc_data = reg_obj.lbl_encoding()
    reg_obj.train_test_split()
    reg_obj.lin_reg_process()
