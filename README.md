# Federated Fitness Recommendation System
### Privacy-Preserving Personalized Workout Recommendations using Federated Learning

A **production-ready Federated Learning system** that delivers personalized workout recommendations while **preserving user privacy**.  
The project combines **Federated Learning, Differential Privacy, Secure Aggregation, and Explainable AI (SHAP)** on realistic, non-IID mobile fitness data.

---

## 🔍 Problem Statement

Traditional fitness recommendation systems require **centralizing sensitive health data**, which introduces:
- Privacy risks
- Regulatory concerns (GDPR / HIPAA)
- Lack of user trust

This project solves the problem by **training models directly on user devices** and sharing **only privacy-protected model updates**, never raw data.

---

## 🎯 Solution Overview

- User fitness data **never leaves the client device**
- Each client trains a local model
- Only **differentially-private model updates** are sent to the server
- The server performs **secure aggregation** using Federated Averaging
- Recommendations are made **interpretable using SHAP**

---

## ✨ Key Features

- ✅ Federated Learning using **FedAvg**
- ✅ Differential Privacy (DP-SGD)
- ✅ Secure aggregation of client updates
- ✅ Realistic **non-IID client data**
- ✅ Explainable recommendations with **SHAP**
- ✅ Centralized vs Federated comparison
- ✅ Clean, modular, production-ready codebase

---

## 🏗️ System Architecture

