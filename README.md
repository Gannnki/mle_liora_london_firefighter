# Liora 🔥 London Fire Brigade 

## 📌 Project Overview

This project aims to analyze and predict **fire brigade response and mobilisation times** using historical data from the London Fire Brigade.

We combine structured data analysis with machine learning to:

* Understand factors affecting response time
* Predict response time for new incidents

---

## 👥 Team Members

* Yu
* Laura
* Khoi
* Kilian

---

## 📊 Data Sources

We use two official datasets:

### 1. Incident Records

Contains details about each incident:

* Time, location, incident type
* Property and geographic information

🔗 https://data.london.gov.uk/dataset/london-fire-brigade-incident-records

---

### 2. Mobilisation Records

Contains details of each dispatched fire engine:

* Dispatch, travel, and arrival times
* Station and resource information

🔗 https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records

---

## 🔗 Data Integration

The two datasets are linked via:

> **IncidentNumber**

This creates a **one-to-many relationship**:

* One incident → multiple fire engine mobilisations

---

## 🎯 Project Goals

### 1. Prediction (Core Task)

Use machine learning to predict:

* **AttendanceTimeSeconds (response time)**

This is formulated as a **regression problem**.

---

### 2. Analysis

Identify key factors influencing response time:

---

### 3. 

---

## ⚙️ Methodology

### Step 1: Data Preprocessing

* Filter consistent time range (2021–2024)
* Clean missing values
* Merge datasets using IncidentNumber

---

### Step 2: Feature Engineering

---

### Step 3: Modeling
---

### Step 4: Evaluation

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

---

### Step 5: (Optional) Agent or Workflow automation
---

## 📁 Project Resources

### Documentation

* Project description:
  https://docs.google.com/document/d/1368CKhHYetKFK2qU7VwEnXspXk-ZTGqBtSWGnHBcJ7M

* Methodology:
  https://docs.google.com/document/d/1sbgOhiBA4hIYgkO-wrEDZrAejmoz9Ezr5EEwDqsdGMw

---

### Data Auditing Template

https://docs.google.com/spreadsheets/d/1JI7_DBcSXJl5UxB8VY-Ybyr3T1NdK0PCDkO1t-ZRjj8

---

## 🧠 Key Insight

* The problem is fundamentally a **regression task**
* Avoid data leakage by excluding arrival timestamps and derived response variables
* Combine incident context + mobilisation process for prediction

---

## 🚀 Future Work
tbd
