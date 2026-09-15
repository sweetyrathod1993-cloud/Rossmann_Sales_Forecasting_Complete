# Task 2.3: Selection and Defense of Loss Function

## Chosen Loss Function: Root Mean Square Percentage Error (RMSPE)
$$\text{RMSPE} = \sqrt{ \frac{1}{N} \sum_{i=1}^N \left( \frac{y_i - \hat{y}_i}{y_i} \right)^2 }$$

## Rationale and Defense:

1. **Scale Independence Across Store Sizes:**
   Rossmann operates 1,115+ diverse stores across several cities. High-volume transit stores (e.g., StoreType 'b') generate daily sales upwards of €20,000–€30,000, while smaller neighborhood stores generate €2,000–€5,000.
   Under standard Euclidean metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE), an absolute prediction error of €1,000 on a large store is penalized equally to an error of €1,000 on a small store, despite representing a negligible 4% relative error for the former and an intolerable 50% forecasting failure for the latter. RMSPE penalizes relative percentage deviation, ensuring fair and balanced optimization across all store tiers.

2. **Managerial Decision Utility:**
   Store managers and supply chain planners order inventory and allocate staffing based on proportional margins. Communicating forecasting accuracy in relative percentage terms (e.g., *our model forecasts with 11.4% average error*) provides transparent, actionable business confidence.

3. **Benchmarking Consistency:**
   RMSPE is the official metric of the Rossmann Store Sales competition, enabling rigorous validation against state-of-the-art retail benchmarks.
