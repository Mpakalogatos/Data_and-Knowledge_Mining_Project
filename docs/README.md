# 🧬 Diabetes Risk Prediction via Data Mining

## 🔬 Project Overview

  This project was developed for the Data Mining and Knowledge Discovery course (7th Semester, Academic Year 2024-2025). The primary objective is to leverage data mining techniques to solve real-world healthcare challenges—specifically, predicting the risk of diabetes    using the PIMA Indian Diabetes Dataset.
  
  By analyzing patient metrics, the project aims to help healthcare providers identify high-risk individuals, leading to more proactive and personalized care.

  ## 📚 Dataset

  The project utilizes the PIMA Indian Diabetes Dataset (sourced from Kaggle), which includes clinical data for thousands of patients.

  Target Variable: Outcome (1 for diabetic, 0 for non-diabetic).
    
  Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age.

  ## 📊 Methodology & Algorithms

  The project implements two primary data mining approaches using Python:

  ### 1. Classification (Decision Trees)

      Script: DT_diabetes.py 

  - Goal: Predict categorical outcomes (diabetic vs. non-diabetic) to suggest preventive measures.

        Process: The data is split into training (70%) and testing (30%) sets. A DecisionTreeClassifier is trained to identify patterns that lead to a diabetes diagnosis.

  ### 2. Clustering (K-Means)

    Script: KMEANS_diabetes.py 
    
  - Goal: Group patients based on biological similarities to uncover hidden patterns in the data.
  
        Process: Features are normalized using StandardScaler before applying the K-Means algorithm with two clusters ($k=2$).

  ## 🌿 Key Insights & Impact

  - Clinical Benefits: Enables personalized treatment plans and improved resource allocation in clinics.
  
  - Operational Efficiency: Helps healthcare administrators reduce costs and optimize patient flow.
  
  - Strategic Decisions: The extracted knowledge supports data-driven prevention strategies and population health management.

  ## ✅ Requirments

  -Python 3.x
  -Pandas
  -NumPy
  -Scikit-learn
  -Matplotlib
