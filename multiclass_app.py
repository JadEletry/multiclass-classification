import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

# Load dataset
df = pd.read_csv("multiclass_data.csv")
X = df[['feature1', 'feature2']].values
y = df['labels'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sidebar model selection
st.title("Multiclass Classification Demo")
model_name = st.sidebar.selectbox("Choose Classifier", [
    "Logistic Regression",
    "Support Vector Machine",
    "Decision Tree",
    "K-Nearest Neighbors"
])

# Model setup
if model_name == "Logistic Regression":
    C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=C, multi_class='ovr', max_iter=200)
elif model_name == "Support Vector Machine":
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    gamma = st.sidebar.selectbox("Kernel Coefficient (gamma)", ["scale", "auto"])
    model = SVC(C=C, kernel='rbf', gamma=gamma)
elif model_name == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 15, 5)
    model = DecisionTreeClassifier(max_depth=max_depth)
else:
    k = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)

# Train model
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Accuracy
acc = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {acc:.2f}")

# Confusion matrix
fig_cm, ax_cm = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
st.pyplot(fig_cm)

# Decision boundary plot
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3)
scatter = ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolor='k', cmap='tab10')
legend_labels = np.unique(y_train)
legend_handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=scatter.cmap(scatter.norm(i))) for i in legend_labels]
ax.legend(legend_handles, legend_labels, title="Classes")
ax.set_title(f"Decision Boundary - {model_name}")
ax.set_xlabel("Feature 1 (scaled)")
ax.set_ylabel("Feature 2 (scaled)")
st.pyplot(fig)
