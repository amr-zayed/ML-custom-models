import numpy as np
class CustomLinearRegression():
    def __init__(self, LearningRate, IterationNumber):
        self.LR = LearningRate
        self.NoIter = IterationNumber
        self.regularization = 0
        self.alpha = 1

    def predict(self, X):
        # return np.dot(self.W, X)+self.bias
        return self.W * X+self.b
    
    def fit(self, X, Y, regType=None, alpha=1):
        self.SamplesCount, self.FeaturesCount = X.shape
        self.W = np.zeros(self.FeaturesCount)
        self.b = 0
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.alpha = alpha
        self.regType = regType
        for _ in range(self.NoIter):
            self.update()

    def update(self):
        WDrivitive = None
        BDrivitive = None
        BDrivitiveArr = None
        WDrivitiveArr = None

        wx = (self.W*self.X).sum(axis=1)
        wx.shape = (self.SamplesCount,1)
        commonCalc = self.Y-((self.W*self.X)+self.b)
        if self.regType=='ridge':
            self.regularization = self.W.sum()
        elif self.regType=='lasso':
            self.regularization = np.divide(self.W,np.abs(self.W), out=np.ones_like(self.W), where=self.W!=0).sum()
        WDrivitiveArr = commonCalc*(-self.X)
        WDrivitive = (WDrivitiveArr.sum(axis=0))/442 + self.alpha*self.regularization
        self.W = self.W-self.LR*WDrivitive
        
        BDrivitiveArr = commonCalc*(-1)
        BDrivitive = BDrivitiveArr.sum()/442
        self.b = self.b-self.LR*BDrivitive
        
            
        
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:,2].reshape(442,1)
diabetes_Y = diabetes_y[:].reshape((442,1))



reg= CustomLinearRegression(1.4,1000)
reg.fit(diabetes_X,diabetes_Y)

regr = LinearRegression()
regr.fit(diabetes_X,diabetes_Y)
x_predict = np.linspace(-0.2,0.2,5).reshape(5,1)

y_predict = reg.predict(x_predict)
yr_predict = regr.predict(x_predict)
plt.scatter(diabetes_X, diabetes_Y, color="black")

#custom built linear regression model
plt.plot(x_predict, y_predict, color="blue", linewidth=3)
#sklearn built-in linear regression model
plt.plot(x_predict, yr_predict, color="red", linewidth=3)
plt.show()