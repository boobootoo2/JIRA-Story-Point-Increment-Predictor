# 🧮 JIRA Story Point Increment Predictor

<p align="center">
  <img src="https://github.com/boobootoo2/JIRA-Story-Point-Increment-Predictor/blob/main/story-point-increment-predictor.png?raw=true" 
       alt="JIRA Story Point Predictor Workflow" width="40%">
</p>

This project demonstrates how to train and deploy a **Machine Learning model** that predicts **JIRA story point increments** based on issue summaries and descriptions.

The workflow leverages **Hugging Face embeddings** (`sentence-transformers/all-MiniLM-L6-v2`) and **Databricks AutoML** to train an `XGBoostRegressor` model capable of estimating story point changes.

---

## 🧠 Overview

This repository includes a public dataset and notebooks used to build a **JIRA Story Point Increment Predictor**.  
Each JIRA issue’s **summary** and **description** text fields are converted into **semantic embeddings**, and the resulting vectors are used to predict the *incremental story point value*.

**Model Stack:**
- 🧩 **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- ⚙️ **Platform:** Databricks AutoML
- 📈 **Algorithm:** XGBoostRegressor
- ☁️ **Language:** Python

---

## 💡 Why This Matters

JIRA estimation can be subjective and time-consuming, especially under tight delivery schedules.  
This solution provides a **data-driven baseline** for sizing issues, using a **generalized 8-increment complexity scale**.

> Teams can map these predicted increments to their own story point or hour scales — giving a consistent, automated way to gauge issue complexity.

---

## 🔗 Integration Concept

When hosted behind an **API endpoint**, this model can easily integrate with JIRA or other client systems.

**Flow:**
1. Client sends `summary` and `description` to the API.
2. API generates embeddings via the sentence transformer.
3. Model returns a **predicted increment value**.
4. Client maps that increment to their own story point scale.

**Example Response:**
```json
{
  "summary": "Add user authentication to onboarding flow",
  "predicted_increment": 5
}
