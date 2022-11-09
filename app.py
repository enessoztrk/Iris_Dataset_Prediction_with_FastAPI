#########################################
# Iris Dataset Prediction with Fast API #
#########################################

# Libraries
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from fastapi.templating import Jinja2Templates
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from fastapi import FastAPI, Request
import matplotlib.pyplot as plt
from joblib import dump, load
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np
import uvicorn


# Load Dataset
dff = load_iris()

features = dff.data

labels = dff.target

# labelsname = df.target_names
labelsname = list(dff.target_names)

featuresName = dff.feature_names

[labelsname[label] for label in labels[:3]]

type(features)
# numpy.ndarray

df = pd.DataFrame(features)
type(df)

df.columns = featuresName


# Data Analysis
def check_df(df, head=5):
    print("##################### Shape #####################")
    print(df.shape)

    print("##################### Types #####################")
    print(df.dtypes)

    print("##################### Head #####################")
    print(df.head(head))

    print("##################### Tail #####################")
    print(df.tail(head))

    print("##################### NA #####################")
    print(df.isnull().sum())

    print("##################### Quantiles #####################")
    print(df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    
check_df(df)

def cat_summary(df, col_name, plot=False):

    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(df)}))

    if plot:
        sns.countplot(x=df[col_name], data=df)
        plt.show()
        
cat_summary(df, "sepal length (cm)", plot=False)


# Data Visualization
ax = px.scatter_3d(df, x="petal width (cm)", y="petal length (cm)", z="sepal length (cm)",template= "plotly_dark");
ax.show()

df.plot(kind="box",figsize=(16,9));

df.plot(x ="sepal length (cm)", y="sepal width (cm)",kind="scatter", figsize=(16,9));


# Select Model
clf = KNeighborsClassifier(n_neighbors = 8)


# Split Dataset
X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state =42)


# Train Model
clf.fit(X_train, y_train)
accuracy = clf.score(X_train,y_train)
"accuracy on train data {:.2}%".format(accuracy)


# Test Model
accuracy = clf.score(X_test,y_test)
"accuracy on test data {:.2}%".format(accuracy)


# Save Model
filename = "myFirstSavedModel.joblib"
dump(clf, filename)


# Load Model
clfUploaded = load(filename)
accuracy = clfUploaded.score(X_test,y_test)
"accuracy on test data {:.2}%".format(accuracy)


# FAST API 🎯
templates = Jinja2Templates(directory='templates')

app = FastAPI()

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


@app.get("/predict/")
async def make_prediction(request: Request, L1:float, W1:float, L2:float, W2:float):
    testData = np.array([L1,W1,L2,W2]).reshape(-1,4)
    probalities = clfUploaded.predict_proba(testData)[0]
    predicted = np.argmax(probalities)
    probabilty = probalities[predicted]
    predicted = labelsname[predicted]
    return templates.TemplateResponse("prediction.html", {"request": request,
                                                          "probalities":probalities,
                                                          "predicted": predicted,
                                                          "probabilty":probabilty})

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host="127.0.0.1")
    uvicorn.run(app, port=8000, host="127.0.0.1")
