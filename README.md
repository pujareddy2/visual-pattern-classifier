Visual Pattern Classification (KNN vs SVM)

A machine learning project that classifies manufacturing items based on two image-derived numerical features — Shape and Texture.
This project compares two popular ML algorithms, K-Nearest Neighbors (KNN) and Support Vector Machine (SVM), and visualizes their decision boundaries in an interactive web application built using Streamlit.

🔗 Live Demo: https://visual-pattern-classifier.streamlit.app/

📘 Full Theory (How It Works): https://visualpatternclassification.notion.site/Visual-Pattern-Classification-2ba6df4e0f008099a8a4d3087b085cf0?pvs=74

📌 Project Overview

Manufacturing industries inspect thousands of items each day. Each item has measurable properties such as shape, texture, or pattern. Using these numerical features, we can train ML models to automatically classify items into categories like:

Good

Defective

This project demonstrates:

Feature normalization

Training KNN and SVM classifiers

Visualizing decision boundaries

Comparing classification performance

Deploying the model through an interactive interface

🎯 Objectives

Generate or load 2D feature data (shape & texture).

Normalize the features using StandardScaler.

Train two classification models:

KNN

SVM (RBF Kernel)

Visualize their decision boundaries.

Compare accuracy & misclassification counts.

Build a modern, minimal, and interactive Streamlit web application.

Deploy the app online for real-time use.

🧠 Machine Learning Concepts Used
1. Feature Normalization (Standardization)

To ensure both features have equal importance:

z = (x − mean) / standard deviation


Normalization is crucial because KNN and SVM depend on distance and margin.

2. K-Nearest Neighbors (KNN)

Classifies based on the majority vote of nearest neighbors.

Uses Euclidean distance for comparison.

Simple and intuitive but can be sensitive to noise.

Produces flexible, irregular decision boundaries.

3. Support Vector Machine (SVM)

Finds the optimal separating boundary (maximum margin).

Uses RBF kernel for non-linear separation.

More stable and generalizable than KNN.

Produces smooth decision boundaries.

🛠️ Technologies Used

Python

Streamlit (Web UI)

NumPy & Matplotlib (Visualization)

Scikit-Learn

StandardScaler

KNeighborsClassifier

SVC

make_classification

train_test_split

📊 Workflow Diagram
Data Generation/Loading → Normalization → Train-Test Split → 
Train KNN & SVM Models → Decision Boundary Visualization → 
Prediction → Streamlit Web App → Deployment

🚀 How to Run Locally
1. Clone this repository
git clone https://github.com/your-username/visual-pattern-classifier.git
cd visual-pattern-classifier

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py


The app will open in your browser.

📷 Screenshots

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e58ba1db-f128-4f31-baf4-134ddbe9f3f2" />


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/44d9e390-35cc-4c9d-a0df-2ea6420c3143" />


📈 Results
Model	Accuracy	Misclassifications
KNN	0.95–1.00	Few
SVM	0.96–1.00	Often fewer

🔎 Observation:
SVM usually performs slightly better and creates smoother boundaries.

🌐 Live Web Application

The project is deployed here:

👉 https://visual-pattern-classifier.streamlit.app/

Features of the app:

Adjustable feature sliders

Choose between KNN or SVM

Real-time prediction

Decision boundary visualization

Minimal dark/light UI

“How It Works” button linking to Notion

📚 Project Structure
├── app.py                # Streamlit Web App
├── requirements.txt      # Python Dependencies
├── README.md             # Documentation
└── screenshots/          # Images for README (optional)

💡 Future Enhancements

Add more features (color, symmetry, edges)

Support multi-class classification

Integrate actual images using CNNs

Provide downloadable predictions

Add model tuning options (C, gamma, k-value slider)


🏁 Conclusion

This project successfully demonstrates:

How machine learning can classify manufacturing items using numerical features

The complete pipeline from data → models → visualization → deployment

Comparison between two widely used algorithms (KNN and SVM)

Real-time classification using a modern web interface
