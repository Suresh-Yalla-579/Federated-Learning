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



## 📊 Dataset

- **Synthetic mobile fitness data**
- Each client contains **50–200 samples**
- Features include:
  - Age
  - Gender
  - Steps per day
  - Heart rate
  - Calories burned
  - Weight & height
- Labels: Workout types (HIIT, Cardio, Strength, Yoga, etc.)
- **Non-IID distribution** to simulate real-world behavior

---

## 🧠 Model Architecture

- **Multi-Layer Perceptron (MLP)**
- Hidden layers: `[128, 64, 32]`
- Activation: ReLU
- Regularization:
  - Dropout (0.3)
  - L2 Weight Decay (0.01)
- Output:
  - Softmax over workout classes

---

## 🔐 Privacy Mechanisms

### Differential Privacy (DP-SGD)

- Gradient clipping (`l2_norm_clip = 1.0`)
- Gaussian noise injection
- Privacy budget:
  - **ε ≈ 10**
  - **δ = 1e-5**

### What this guarantees:
- Individual user data **cannot be inferred**
- Participation of a single client is **indistinguishable**
- Protection against gradient leakage attacks

---

## 📈 Performance Summary

| Model Type | Accuracy | F1 Score | Recall@3 |
|-----------|----------|----------|----------|
| Centralized | 75–85% | 0.75–0.85 | ~0.90 |
| Federated + DP | 70–80% | 0.70–0.80 | ~0.85 |

> Accuracy drops slightly due to privacy noise — a **necessary and expected trade-off**.

---

## 🔎 Explainability (SHAP)

Each recommendation includes a **feature-level explanation**:

Example:
Predicted Workout: HIIT
Confidence: 85%

Top Influential Features:

Steps per day (+)

Resting heart rate (+)

Age (+)

Calories burned (+)

Weight (-)


Outputs:
- SHAP feature importance plots
- Per-prediction explanations
- Stored in `results/plots/`

---

### 📚 References
McMahan et al., Communication-Efficient Learning of Deep Networks from Decentralized Data

Abadi et al., Deep Learning with Differential Privacy

Lundberg & Lee, SHAP: A Unified Approach to Interpreting Predictions

Flower Federated Learning Framework

### 📄 License
This project is intended for educational and research purposes.

### 🤝 Contributions
Contributions are welcome:

Async Federated Learning

Personalization layers

Advanced DP accountants

Real mobile deployment

### 📬 Contact
For questions, suggestions, or collaboration, please open an issue.

Built with privacy-first ML principles 💪





