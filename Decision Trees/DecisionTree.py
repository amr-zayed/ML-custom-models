import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True)

class Node():
    def __init__(self, data, parent=None):
        self.data=data
        self.parent=parent
        self.Children=[]

    def addData(self, data):
        self.data.update(data)

    def setData(self, data):
        self.data=data

    def addChild(self, node):
        self.Children.append(node)

    def predict(self, x):
        for child in self.Children:
            if len(child.Children)==0:
                return child.data
            if x[self.data['nxtCol']]==child.data['value']:
                return child.predict(x)
        return self.Children[0].Children[0].data
class DecisionTree():
    def __init__(self):
        self.root=None
        pass

    def fit(self, x,y):
        self.X = x
        self.Y = y
        # rows, columns = x.shape
        stack= []
        self.XTemp = x.copy()
        self.YTemp = y.copy()
        self.root = Node({})
        self.fitRec(self.root, stack)

    def fitRec(self, node, stack):
        featureList = np.hstack([self.X, self.Y])
        for (val, index) in stack:
            featureList = featureList[featureList[:, index]==val]

        if featureList.shape[0]<=1 or len(stack)==featureList.shape[1]:
            classes, counts = np.unique(featureList[:,-1], return_counts=True)            
            results = np.array(list(zip(classes, counts)))
            prediction = results[results[:,1]==results[:,1].max()][0,0]
            node.setData(prediction)
            return

        rows, columns = featureList.shape
        maxIG = -100
        maxIndex=0
        for feature in range(columns-1):
            IG = self.calculateIG(np.hstack([featureList[:,feature].reshape(rows,1), featureList[:,-1].reshape(rows,1)]))
            if maxIG<IG:
                maxIG=IG
                maxIndex=feature
        node.addData({"nxtCol": maxIndex})
        values = np.unique(featureList[:,maxIndex])
        del featureList
        for value in values:
            childNode=Node({"value": value}, node)
            node.addChild(childNode)
            stack.append((value, maxIndex))
            self.fitRec(childNode, stack)
            stack.pop()
        pass


    def calculateIG(self, feature):
        values, featureCounts = np.unique(feature[:,0], return_counts=True)
        
        def calculateEntropy(arr):
            _, counts = np.unique(arr, return_counts=True)
            denom = counts.sum()
            probabilities = counts/denom
            return -(probabilities*np.log(probabilities)).sum()
        
        Es = calculateEntropy(feature[:,1])
        Ef=[]
        for value in values:
            featureClass = feature[feature[:,0]==value]
            Ef.append(calculateEntropy(featureClass[:,1]))
        sizeRatio = featureCounts/featureCounts.sum()
        return (Es - (Ef*sizeRatio).sum())

    def predict(self, x):
        predictions = []
        for row in x:
            predictions.append(self.root.predict(row))
        predictions = np.array(predictions)
        return predictions
    
df=pd.read_csv('cardio_train.csv', sep=';', header=0)
data = df.values
x = data[:, 0:-1]
x = np.delete(x, [0, 1, 3, 4, 5, 6], 1)
y = data[:, -1].reshape((70000,1))

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

customModel = DecisionTree()
scikitModel = tree.DecisionTreeClassifier(criterion="entropy")

scikitModel.fit(X_train, y_train)
customModel.fit(X_train, y_train)
customPrediction = customModel.predict(X_test)
scikitPrediction= scikitModel.predict(X_test)
