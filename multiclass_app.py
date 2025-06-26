import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Load and prepare data
data = pd.read_csv("multiclass_data.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Classifier options
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Streamlit UI
st.title("Multiclass Classification Demo")
classifier_name = st.sidebar.selectbox("Choose Classifier", list(classifiers.keys()))
show_cm = st.sidebar.checkbox("Show Confusion Matrix", value=True)
show_report = st.sidebar.checkbox("Show Classification Report", value=True)

# Train and predict
clf = classifiers[classifier_name]
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = np.mean(y_pred == y_test)
st.write(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
if show_cm:
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax)
    st.pyplot(fig)

# Classification report
if show_report:
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("Precision / Recall / F1 Score:")
    st.dataframe(pd.DataFrame(report).transpose())

# Decision boundary (2D only)
def plot_decision_boundary(X, y, clf, title):
    if X.shape[1] != 2:
        st.warning("Decision boundary plot requires exactly 2 features. Skipping this plot.")
        return

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    try:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        fig, ax = plt.subplots()
        ax.contourf(xx, yy, Z, alpha=0.3)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k')
        ax.set_title(title)
        ax.set_xlabel("Feature 1 (scaled)")
        ax.set_ylabel("Feature 2 (scaled)")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not plot decision boundary: {e}")


plot_decision_boundary(X_test, y_test, clf, f"Decision Boundary - {classifier_name}")
