import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Explore Different Classifiers")


dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris Dataset", "Breast Cancer Dataset", "Wine Quality Dataset"))

clf_name = st.sidebar.selectbox("Select Classifier", ("K-Nearest Neighbors", "Support Vector Machine", "Random Forest Classifier"))


def get_dataset(dataset_name):
    if dataset_name == "Iris Dataset":
        data = datasets.load_iris()
    
    elif dataset_name == "Breast Cancer Dataset":
        data = datasets.load_breast_cancer()

    elif dataset_name == "Wine Quality Dataset":
        data = datasets.load_wine()
    
    X = data.data
    y = data.target

    return X, y


X, y = get_dataset(dataset_name)
st.write("**Shape of Dataset**", X.shape)
st.write("**Number of Classes**", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "K-Nearest Neighbors":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K

    elif clf_name == "Support Vector Machine":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C

    elif clf_name == "Random Forest Classifier":
        max_depth = st.sidebar.slider("Max Depth", 2, 15)
        n_estimators = st.sidebar.slider("N_Estimators", 1, 1000)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(clf_name)

def get_classifier(clf_name, params):
    if clf_name == "K-Nearest Neighbors":
        clf = KNeighborsClassifier(n_neighbors = params["K"], metric = 'minkowski')
    
    elif clf_name == "Support Vector Machine":
        clf = SVC(C = params["C"], kernel = 'rbf',random_state = 0)

    elif clf_name == "Random Forest Classifier":
        clf = RandomForestClassifier(max_depth = params["max_depth"], n_estimators = params["n_estimators"], random_state = 0)
    
    return clf

clf = get_classifier(clf_name, params)


#Classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.write(f"**Classifier** = {clf_name}")
st.write(f"**Accuracy** = {accuracy}")


pca = PCA(2)
X_pca = pca.fit_transform(X)

x1 = X_pca[:, 0]
x2 = X_pca[:, 1]

plt.figure()
plt.scatter(x1, x2, c = y, cmap = "viridis", alpha = 0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()

st.pyplot()