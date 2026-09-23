# 💳 AI Finance Intelligence & Fraud Detection System

An end-to-end Machine Learning application for detecting potentially fraudulent financial transactions and assessing transaction risk in real time.

The system combines Machine Learning, FastAPI, Streamlit, Docker, and data analytics into a complete deployable fraud detection solution.

---

## 🚀 Live Demo

### 🖥️ Live Dashboard
https://ai-finance-intelligence-fraud-detection-system-yyvvaop6jxa9nf8.streamlit.app/

### ⚡ FastAPI Backend
https://finance-fraud-api.onrender.com

### 📚 API Documentation
https://finance-fraud-api.onrender.com/docs

---

## 📌 Project Overview

Financial fraud detection requires identifying unusual transaction patterns while minimizing false alarms.

This project uses Machine Learning to analyze transaction attributes such as:

- Transaction amount
- Transaction frequency
- Merchant risk
- Customer information
- Device information
- International transactions
- Online transactions
- Failed authentication attempts
- Distance from home/usual location
- Recent account activity
- Amount deviation from normal spending

The system generates a fraud probability and classifies the transaction into a risk level.

---

## 🎯 Objectives

- Detect potentially fraudulent transactions using Machine Learning
- Calculate fraud probability for individual transactions
- Classify transactions into LOW, MEDIUM, and HIGH risk
- Provide an interactive analytics dashboard
- Expose predictions through a REST API
- Containerize the application using Docker
- Deploy the backend and frontend online

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │       User          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    │     Frontend        │
                    └──────────┬──────────┘
                               │
                         HTTP POST
                               │
                               ▼
                    ┌─────────────────────┐
                    │     FastAPI API     │
                    │       Backend       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ ML Fraud Detection  │
                    │       Model         │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌───────────────────────────┐
                 │ Fraud Probability + Risk │
                 │          Level            │
                 └───────────────────────────┘