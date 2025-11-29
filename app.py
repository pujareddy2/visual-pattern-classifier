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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# GLOBAL CSS
# ----------------------------------------------------------
st.markdown("""
    <style>

    .big-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: -5px;
    }

    .sub-text {
        text-align: center;
        font-size: 17px;
        margin-bottom: 20px;
        opacity: 0.85;
    }

    .how-btn {
        background: #4B7BEC;
        color: white !important;
        padding: 12px 30px;
        border-radius: 8px;
        font-size: 16px;
        text-decoration: none;
        font-weight: 500;
    }
    .how-btn:hover {
        background: #3a63c9;
        transition: 0.2s;
    }

    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# HEADER
# ----------------------------------------------------------
st.markdown("<div class='big-title'>Visual Pattern Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Visual pattern classifying 2D feature patterns using KNN and SVM.</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ----------------------------------------------------------
# HOW IT WORKS BUTTON
# ----------------------------------------------------------
colA, colB, colC = st.columns([1,2,1])
with colB:
    st.markdown("""
        <div style="text-align:center; margin-top: -10px;">
            <a class="how-btn" target="_blank"
               href="https://www.notion.so/visualpatternclassification/Visual-Pattern-Classification-2ba6df4e0f008099a8a4d3087b085cf0?source=copy_link">
               How It Works
            </a>
        </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# SIDEBAR INPUTS
# ----------------------------------------------------------
st.sidebar.header("Input Features")
shape = st.sidebar.number_input("Shape Feature", value=1.0, step=0.1)
texture = st.sidebar.number_input("Texture Feature", value=1.0, step=0.1)
model_choice = st.sidebar.selectbox("Choose Model", ["KNN", "SVM"])
predict_btn = st.sidebar.button("Predict")

# ----------------------------------------------------------
# DATA GENERATION
# ----------------------------------------------------------
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    class_sep=1.5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

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

# ACCURACY VALUES
acc_knn = accuracy_score(y_test, knn.predict(X_test_scaled))
acc_svm = accuracy_score(y_test, svm.predict(X_test_scaled))

# ----------------------------------------------------------
# LAYOUT: LEFT (Prediction) — RIGHT (Decision Boundary)
# ----------------------------------------------------------
col1, col2 = st.columns([1,2])

with col1:
    st.markdown("<h3>Prediction</h3>", unsafe_allow_html=True)

    if predict_btn:
        user_input = np.array([[shape, texture]])
        scaled_input = scaler.transform(user_input)

        if model_choice == "KNN":
            pred = knn.predict(scaled_input)[0]
            st.success(f"Predicted Class (KNN): {pred}")
        else:
            pred = svm.predict(scaled_input)[0]
            st.success(f"Predicted Class (SVM): {pred}")
    else:
        st.info("Adjust values in the sidebar and click Predict")

    st.markdown("<h3 style='margin-top:25px;'>Model Accuracy</h3>", unsafe_allow_html=True)
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

    fig, ax = plt.subplots(figsize=(7,6))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="coolwarm", s=40)
    ax.set_title(title)

    st.pyplot(fig)

with col2:
    st.markdown(f"<h3>{model_choice} Decision Boundary</h3>", unsafe_allow_html=True)

    if model_choice == "KNN":
        plot_decision_boundary(knn, X_train_scaled, y_train, "KNN Decision Boundary")
    else:
        plot_decision_boundary(svm, X_train_scaled, y_train, "SVM Decision Boundary")

# ----------------------------------------------------------
# COMPARISON CHART (BAR GRAPH)
# ----------------------------------------------------------
st.markdown("<h3>Accuracy Comparison Chart</h3>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(5,4))
models = ["KNN", "SVM"]
accuracies = [acc_knn, acc_svm]

ax.bar(models, accuracies, color=["#4B7BEC", "#26de81"])
ax.set_ylim(0, 1.1)
ax.set_ylabel("Accuracy")
ax.set_title("KNN vs SVM Accuracy Comparison")

st.pyplot(fig)

# ----------------------------------------------------------
# MISCLASSIFIED COUNT
# ----------------------------------------------------------
st.markdown("<h3>Misclassified Samples</h3>", unsafe_allow_html=True)
mis_knn = sum(knn.predict(X_test_scaled) != y_test)
mis_svm = sum(svm.predict(X_test_scaled) != y_test)

st.write(f"KNN Misclassified Samples: **{mis_knn}**")
st.write(f"SVM Misclassified Samples: **{mis_svm}**")

