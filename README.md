# coffee-shop-loyalty-final
# Coffee Shop Loyalty: Final Exam (AI Fundamentals)

**Variant 16: Coffee shop loyalty** · Author: Togzhan Akhmediyeva

Two projects on the same synthetic dataset of 1,000 coffee-shop customers
(features: visits per week, order price, wait time):

- **Module 9, Supervised Learning:** predict whether a customer returns next week,
  then turn the probability into a cost-based coupon decision.
- **Module 10, Unsupervised Learning:** find customer groups (K-Means, DBSCAN),
  reduce dimensions (PCA, t-SNE, UMAP) and detect unusual customers (Isolation Forest).

## Files

| File | Description |
|---|---|
| `Practice_8_Akhmediyeva_Togzhan_Coffee_shop_loyalty_with_references.ipynb` | Module 9 notebook (code, results, references) |
| `Final_Project_P2_Variant16_Coffee_shop_loyalty_with_references.ipynb` | Module 10 notebook (code, results, references) |
| `Final_Project_Module9_Report_Variant16.docx` | Module 9 report |
| `Final_Project_Module10_Report_Variant16.docx` | Module 10 report |
| `akhmediyeva togzhan.pdf` | Presentation slides |

## How to run

1. Open a notebook in Google Colab (File > Upload notebook).
2. Choose Runtime > Run all.
3. The Module 10 notebook installs `umap-learn` in one cell; no other setup is needed.

All random steps use seed 42, so the results are reproducible.

## Key results

**Module 9:** Logistic Regression and Random Forest have almost equal cross-validated F1
(0.695 vs 0.693), so the simpler model is used. Test accuracy 0.67, ROC-AUC 0.78.
A cost-based coupon threshold (about 0.35) is the only profitable policy (+$5.50 on the test set).

**Module 10:** K-Means suggests K = 4 (silhouette 0.27); DBSCAN finds one dense cluster and
50 noise points; two PCA components keep 68% of the variance; Isolation Forest flags 50
customers, 38 of which are also DBSCAN noise. Frequent visitors return 84% of the time,
long-wait customers 28%.

## Data note

The suggested UCI Online Retail dataset has no wait-time feature, so the data are
synthetic and generated in the notebooks. The results describe the simulation,
not real customers.

## References

Full reference lists are at the end of each notebook (scikit-learn, NumPy, pandas, Matplotlib,
and the original papers for the methods used).
