# Advanced Customer Segmentation Using Unsupervised Learning

## Project Overview
This project implements an end-to-end customer segmentation system using unsupervised machine learning techniques. The objective is to identify hidden behavioral patterns within retail transaction data and convert them into actionable business insights.

Since no labeled customer categories were available, clustering algorithms were applied to discover meaningful segments for strategic marketing and revenue optimization.

---

## Problem Statement
Organizations often possess large volumes of transactional data but lack structured customer segmentation. This limits their ability to:

- Identify high-value customers
- Detect churn-risk customers
- Optimize marketing strategies
- Improve revenue allocation

This project applies unsupervised learning to automatically discover customer groups and provide business-focused recommendations.

---

## Dataset Description
Dataset Used: Online Retail Dataset  
- 5000+ transaction records  
- Customer-level behavioral aggregation  
- RFM (Recency, Frequency, Monetary) feature engineering applied  

Engineered Features:
- Recency
- Frequency
- Monetary
- Average Order Value
- Product Diversity

---

## Algorithms Implemented
The following clustering algorithms were implemented and compared:

- KMeans
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Model (GMM)

---

## Cluster Optimization Techniques
- Elbow Method
- Silhouette Score
- Davies-Bouldin Index

Optimal clusters selected: **3**

Best performing model (technical metric): **GMM (Highest Silhouette Score: 0.3907)**

---

## Key Results

### Customer Segments Identified

1. **Dormant / At-Risk Customers**
   - High Recency
   - Low Frequency
   - Low Revenue Contribution
   - Strategy: Re-engagement campaigns

2. **Premium Loyal Customers**
   - Low Recency
   - High Frequency
   - Highest Revenue Contribution
   - Strategy: Loyalty programs & exclusive offers

3. **Mid-Tier Growth Customers**
   - Moderate purchasing behavior
   - Stable revenue generation
   - Strategy: Upselling & cross-selling

---

## Business Insights

- Premium segment drives majority revenue.
- Dormant segment represents churn risk.
- Targeted segmentation improves marketing ROI.
- Data-driven segmentation supports strategic decision-making.

---
