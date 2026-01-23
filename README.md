# 🧠 Traditional Machine Learning Guide (From Scratch)

> **"What I cannot create, I do not understand." — Richard Feynman**

This repository is a collection of fundamental Machine Learning algorithms implemented entirely from scratch in **Python**, without using high-level libraries like Scikit-Learn or PyTorch for the core logic. 

Each implementation is accompanied by my **handwritten notes**, where I derive the mathematics, gradients, and optimization rules behind the code.

## 🚀 Why This Project?
We often rely on `model.fit()` without understanding the underlying mechanics. My goal with this project was to open the "black box" of Machine Learning.

* **No Black Boxes:** Every algorithm is built using only `NumPy`.
* **Math First:** I derived the equations manually before writing a single line of code.
* **Vectorized:** Implementations are optimized using matrix operations for performance.

## 📂 Repository Structure

| Algorithm | Type | Description | Content |
| :--- | :--- | :--- | :--- |
| **[Linear Regression](./linear_regression)** | Regression | Predicting continuous values using Gradient Descent. | 🐍 Code + 📝 Notes |
| **[Logistic Regression](./logistic_regression)** | Classification | Binary classification using Sigmoid functions. | 🐍 Code + 📝 Notes |
| **[SVM (Support Vector Machine)](./SVM)** | Classification | Finding the optimal hyperplane using Hinge Loss. | 🐍 Code + 📝 Notes |
| **[K-Nearest Neighbors (KNN)](./KNN)** | Classification | Instance-based learning using Euclidean distance. | 🐍 Code + 📝 Notes |
| **[Naive Bayes](./NaiveBayes)** | Classification | Probabilistic classifier based on Bayes' Theorem. | 🐍 Code + 📝 Notes |
| **[Decision Tree](./DecisionTree)** | Classification | Splitting data based on Entropy and Information Gain. | 🐍 Code + 📝 Notes |
| **[Random Forest](./Random%20Forest)** | Ensemble | Bagging multiple Decision Trees for better accuracy. | 🐍 Code + 📝 Notes |
| **[K-Means Clustering](./Kmeans)** | Clustering | Unsupervised partitioning of data into K clusters. | 🐍 Code + 📝 Notes |

## 📝 About the Handwritten Notes
Inside each folder, you will find raw, handwritten notes that cover:
* The **Hypothesis Function** used.
* The **Cost Function** derivation.
* **Gradient Descent** update rules.
* Matrix dimensions and vectorization logic.

## 🛠️ How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/Siva20053/traditionalML-guide.git](https://github.com/Siva20053/traditionalML-guide.git)
    cd traditionalML-guide
    ```
2.  Navigate to an algorithm folder (e.g., Linear Regression):
    ```bash
    cd linear_regression
    ```
3.  Run the Python script:
    ```bash
    python linear_regression.py
    ```

## 📦 Dependencies
* `numpy`
* `matplotlib` (for visualization)

## 🤝 Connect
If you found this helpful or have suggestions on how to optimize the code further, feel free to reach out!

* **GitHub:** [Siva20053](https://github.com/Siva20053)
* **LinkedIn:** [siva9963](https://www.linkedin.com/in/siva9963)

---
*If you find this repo useful, please give it a ⭐!*
