# 💊 Fake Drug Text/Barcode Checker

## AI-Assisted Medicine Verification Prototype

An AI/ML-powered prototype designed to assess supplied medicine information using drug text, NAFDAC reference data, regulatory-alert data, barcode recognition, and machine-learning classification.

> ⚠️ **Prototype Disclaimer:** This system is a demonstration prototype. It does not replace official NAFDAC verification, laboratory testing, or professional pharmaceutical advice.

---

## 🎯 Project Overview

Counterfeit, falsified, substandard, and improperly labelled medicines can pose serious risks to public health.

The Fake Drug Text/Barcode Checker explores how artificial intelligence, machine learning, barcode recognition, and regulatory data can be combined to support preliminary medicine verification.

The system allows a user to provide:

- Drug name
- NAFDAC number
- Batch number
- Manufacturer
- Barcode
- Barcode image

The application then checks the supplied information against available reference and regulatory-alert datasets and uses a machine-learning model to provide an additional assessment.

---

## 🚀 Key Features

### 🔴 NAFDAC Regulatory Alert Detection

Checks whether the supplied drug information or NAFDAC number matches a record in the loaded regulatory-alert dataset.

When a match is found, the system displays:

- Product name
- Alert date
- Classification
- NAFDAC number
- Batch number
- Manufacturer
- Alert reference

---

### 🟢 NAFDAC Reference Matching

Checks supplied drug information against the loaded NAFDAC reference dataset.

The system can display:

- Drug name
- NAFDAC number
- Batch
- Manufacturer
- Barcode
- Registration/reference status

---

### 📦 Barcode Recognition

Users can:

1. Enter a barcode manually, or
2. Upload a barcode image.

The system uses barcode recognition to extract the barcode number and search the available reference database.

---

### 🤖 Machine Learning Assessment

The prototype includes a trained machine-learning model that evaluates selected drug-information features.

The model considers features including:

- NAFDAC information availability
- Batch information availability
- Barcode information availability
- Manufacturer information
- Completeness of supplied drug information

The ML result is presented as a prototype assessment and must not be interpreted as proof that a medicine is genuine or counterfeit.

---

### 🟢 Prototype Reference Database

The system also contains a demonstration reference database for testing the application's matching functionality.

---

## 🧠 System Verification Flow

The prototype uses the following verification priority:

1. 🔴 NAFDAC Regulatory Alert
2. 🟢 NAFDAC Reference Match
3. 🟢 Prototype Reference Match
4. 🔴 Potentially Suspicious
5. 🟠 Needs Further Verification

This prioritizes known regulatory alerts before relying on the machine-learning assessment.

---

## 🛠️ Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- OpenCV
- PyZbar
- Joblib
- CSV datasets
- Machine Learning

---

## 📁 Project Structure

```text
Fake_Drug_Text_Barcode_Checker/
│
├── app.py
├── generate_barcode.py
├── train_model.py
├── README.md
│
├── data/
│   ├── drug_data.csv
│   ├── reference_products.csv
│   ├── nafdac_data.csv
│   └── nafdac_alerts.csv
│
├── model/
│   └── drug_checker_model.pkl
│
└── venv/