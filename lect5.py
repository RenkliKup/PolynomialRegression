import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,PolynomialFeatures
from sklearn.linear_model import LinearRegression

class reg_:
    def __init__(self,path,feauture_cor,output_cor,degree):
        #define degree 
        self.degree=degree
        #define csv
        self.df=pd.read_csv(str(path))
        #define feature coordinate
        self.feauture_cor=feauture_cor
        #define output coordinate
        self.output_cor=output_cor
        #define feature dataframe
        self.feauture1=self.df.iloc[:,[self.feauture_cor]]
        #define output dataframe
        self.output1=self.df.iloc[:,[self.output_cor]]
        #Linear Regression
    def regression(self):
        self.lr=LinearRegression()
        self.lr.fit(self.feauture,self.output)
        #matplotlib
    def plt(self):
        plt.scatter(self.feauture1,self.output1)
        plt.plot(self.feauture1,self.lr.predict(self.feauture))  
        plt.xlabel("output")
        plt.ylabel("feauture")
        plt.title("reg")
        plt.show()
        #Polynomial reg examp 1,2,3 degree
    def howMuchDegree(self):
        self.polynomial_reg=PolynomialFeatures(degree=self.degree)
        self.feauture=self.polynomial_reg.fit_transform(self.df.iloc[:,self.feauture_cor].values.reshape(-1,1))
        self.output=self.df.iloc[:,self.output_cor].values.reshape(-1,1)
        self.regression()
        self.plt()

if __name__=="__main__":
    #examp csv
    c=reg_("maaslar.csv",1,2,2)
    #examp define degree
    c.degree=4
    c.howMuchDegree()
    #examp predict 12 
    print(c.lr.predict(c.polynomial_reg.fit_transform(np.array([12]).reshape(-1,1))))
    
    




