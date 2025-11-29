import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Visual Pattern Classifier",
    layout="wide"
)

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#222;'>Visual Pattern Classifier</h1>
    <p style='text-align:center; font-size:17px; color:#555;'>
        A minimal and professional web application for classifying 2D feature patterns using KNN and SVM.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ----------------------------------------------------------
# HOW IT WORKS BUTTON – Notion Redirect
# ----------------------------------------------------------
colA, colB, colC = st.columns([1,2,1])

with colB:
    st.markdown(
        """
        <div style="text-align:center; margin-top: -10px;">
            <a href="https://www.notion.so/visualpatternclassification/Visual-Pattern-Classification-2ba6df4e0f008099a8a4d3087b085cf0?source=copy_link" 
               target="_blank" 
               style="
                    text-decoration:none; 
                    padding:12px 28px; 
                    background:#4B7BEC; 
                    color:white; 
                    border-radius:6px; 
                    font-size:16px;
                    font-weight:500;
                    letter-spacing:0.3px;
                ">
                How It Works
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------------------------------------
# SIDEBAR INPUTS
# ----------------------------------------------------------
st.sidebar.header("Input Features")
shape = st.sidebar.number_input("Shape Feature", value=1.0, step=0.1)
texture = st.sidebar.number_input("Texture Feature", value=1.0, step=0.1)
model_choice = st.sidebar.selectbox("Choose Model", ["KNN", "SVM"])
predict_btn = st.sidebar.button("Predict")

# ----------------------------------------------------------
# GENERATE DATASET
# ----------------------------------------------------------
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    class_sep=1.5,
    random_state=42
)

# TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# NORMALIZATION
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------------
# TRAIN MODELS
# ----------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train_scaled, y_train)

# ----------------------------------------------------------
# PREDICTION + ACCURACY
# ----------------------------------------------------------
col1, col2 = st.columns([1,2])

with col1:
    st.subheader("Prediction")

    if predict_btn:
        user_input = np.array([[shape, texture]])
        user_input_scaled = scaler.transform(user_input)

        if model_choice == "KNN":
            pred = knn.predict(user_input_scaled)[0]
            st.success(f"Predicted Class (KNN): {pred}")
        else:
            pred = svm.predict(user_input_scaled)[0]
            st.success(f"Predicted Class (SVM): {pred}")
    else:
        st.info("Enter values and click Predict")

    # Accuracy
    st.subheader("Model Accuracy")
    acc_knn = accuracy_score(y_test, knn.predict(X_test_scaled))
    acc_svm = accuracy_score(y_test, svm.predict(X_test_scaled))

    st.write(f"KNN Accuracy: **{acc_knn:.2f}**")
    st.write(f"SVM Accuracy: **{acc_svm:.2f}**")

# ----------------------------------------------------------
# DECISION BOUNDARY PLOT
# ----------------------------------------------------------
def plot_decision_boundary(clf, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="coolwarm", s=40)
    plt.title(title)
    st.pyplot(plt)

with col2:
    st.subheader(f"{model_choice} Decision Boundary")

    if model_choice == "KNN":
        plot_decision_boundary(knn, X_train_scaled, y_train, "KNN Decision Boundary")
    else:
        plot_decision_boundary(svm, X_train_scaled, y_train, "SVM Decision Boundary")

# ----------------------------------------------------------
# MISCLASSIFIED SAMPLES
# ----------------------------------------------------------
st.subheader("Misclassified Samples")

mis_knn = sum(knn.predict(X_test_scaled) != y_test)
mis_svm = sum(svm.predict(X_test_scaled) != y_test)

st.write(f"KNN Misclassified Samples: **{mis_knn}**")
st.write(f"SVM Misclassified Samples: **{mis_svm}**")
