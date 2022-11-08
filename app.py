from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
from fastapi import FastAPI, Request
import uvicorn
from fastapi.templating import Jinja2Templates

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

clf = KNeighborsClassifier(n_neighbors = 8)

X = features
y = labels

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state =42)

clf.fit(X_train, y_train)
accuracy = clf.score(X_train,y_train)
"accuracy on train data {:.2}%".format(accuracy)

accuracy = clf.score(X_test,y_test)
"accuracy on test data {:.2}%".format(accuracy)

filename = "myFirstSavedModel.joblib"
dump(clf, filename)

clfUploaded = load(filename)

accuracy = clfUploaded.score(X_test,y_test)

################################

templates = Jinja2Templates(directory='templates')

app = FastAPI()
labelsname = list(dff.target_names)
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