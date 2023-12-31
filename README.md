### Representation Learning using Multi-Layer Perceptrons (MLPs)

The study utilized multi-layer perceptrons (MLPs) for their versatility with diverse data types and relationships. MLPs with 50 and 100 nodes in hidden layers were chosen to balance complexity and efficiency, considering the feature sizes of datasets like the German Credit Dataset (27 features), Credit Defaulter Dataset (13 features), and Bail Dataset (18 features). This aimed to capture patterns effectively in datasets with 1,000 to 30,000 entries and prevent overfitting in smaller datasets. More details about datasets can be found [here](https://github.com/chirag126/nifty).

MLPs were configured with various hidden layer sizes and activation functions, including 'identity', 'logistic', 'tanh', and 'relu', ranging from simple to complex structures. Hyperparameters like learning rate and batch size added complexity and affected training dynamics and generalization capabilities. The study used default settings for these parameters to minimize variability.

Dimensionality reduction techniques like PCA and t-SNE were implemented to manage high-dimensional data. PCA transformed features into principal components, preserving essential information, while t-SNE retained local data structures in reduced dimensions. PCA was iteratively applied to explore the impact of dimensionality reduction on model performance. t-SNE, limited to three components for efficiency, was used to assess its effect on MLP performance.

The study showed variations in MLP performance across datasets. For instance, an MLP with a single 100-node layer achieved an AUROC of 76.02% and an F1-score of 80.28% on the German Credit Dataset. Fairness metrics like $\Delta_{SP}$ and $\Delta_{EO}$ varied, indicating potential predictive biases. Training MLPs on data with altered sensitive attributes slightly improved fairness metrics but reduced performance, emphasizing the need for techniques balancing fairness and predictive accuracy.

Overall, the research highlighted the importance of careful model configuration and appropriate dimensionality reduction in ensuring fairness and stability in representation learning.

#### Results Table

Here's a summary of the MLP results as compared to other methods:

\* Results obtained after randomly flipping sensitive attributes to augment the data. 

| Dataset | Method | AUROC (↑) | F1-Score (↑) | Unfairness (↓) | Instability (↓) | ΔSP (↓) | ΔEO (↓) |
|---------|------------|-----------|--------------|----------------|-----------------|---------|---------|
| German Credit Dataset | MLP | 76.02 (1.01) | 80.28 (1.88) | 14.64 (14.00) | 28.64 (5.36) | 28.49 (3.21) | 17.65 (2.95) |
| German Credit Dataset | MLP PCA | 54.54 (5.75) | 64.61 (32.36) | 13.84 (21.07) | 16.64 (23.18) | 1.90 (3.01) | 1.28 (1.65) |
| German Credit Dataset | MLP t-SNE | 57.86 (13.85) | 80.06 (2.73) | 4.40 (6.25) | 9.60 (11.00) | 11.40 (13.30) | 7.46 (8.21) |
| German Credit Dataset | MLP* | 73.60 (3.79) | 72.45 (7.11) | 16.64 (15.08) | 29.20 (6.61) | 21.07 (9.13) | 13.05 (7.06) |
| German Credit Dataset | MLP PCA* | 53.41 (4.30) | 53.53 (35.45) | 3.44 (5.76) | 26.32 (30.52) | 1.42 (2.09) | 1.20 (1.51) |
| German Credit Dataset | MLP t-SNE* | 58.79 (15.03) | 80.17 (2.20) | 2.88 (3.75) | 9.20 (10.89) | 9.14 (11.08) | 5.55 (6.61) |
| Credit Defaulter Dataset | MLP | 74.92 (0.12) | 83.50 (0.17) | 1.24 (1.39) | 34.10 (3.23) | 15.78 (0.53) | 13.38 (0.75) |
| Credit Defaulter Dataset | MLP PCA | 74.77 (0.22) | 81.63 (1.88) | 2.47 (1.28) | 35.11 (1.51) | 14.80 (1.56) | 12.61 (1.67) |
| Credit Defaulter Dataset | MLP t-SNE | 74.50 (0.31) | 81.08 (1.97) | 3.37 (1.73) | 31.97 (8.53) | 12.76 (1.93) | 10.82 (1.92) |
| Credit Defaulter Dataset | MLP* | 74.89 (0.20) | 82.90 (0.96) | 2.68 (1.88) | 33.71 (1.17) | 13.90 (0.83) | 11.34 (0.98) |
| Credit Defaulter Dataset | MLP PCA* | 74.63 (0.25) | 82.45 (1.03) | 5.45 (3.42) | 36.18 (0.82) | 12.50 (1.67) | 10.09 (1.57) |
| Credit Defaulter Dataset | MLP t-SNE* | 74.27 (0.37) | 80.53 (2.15) | 6.06 (4.15) | 32.34 (8.10) | 14.12 (1.76) | 12.12 (2.30) |
| Recidivism Dataset | MLP | 94.42 (0.18) | 88.43 (0.27) | 0.45 (0.12) | 42.32 (2.42) | 2.42 (0.28) | 3.57 (0.58) |
| Recidivism Dataset | MLP PCA | 94.42 (0.15) | 88.21 (0.42) | 0.45 (0.10) | 42.53 (2.69) | 2.31 (0.16) | 3.56 (0.26) |
| Recidivism Dataset | MLP t-SNE | 87.61 (4.68) | 76.51 (6.20) | 1.68 (1.48) | 41.55 (1.33) | 6.86 (4.12) | 10.62 (3.15) |
| Recidivism Dataset | MLP* | 95.75 (0.25) | 90.59 (0.30) | 2.51 (0.68) | 45.81 (2.27) | 7.12 (2.14) | 5.68 (3.99) |
| Recidivism Dataset | MLP PCA* | 95.75 (0.22) | 90.47 (0.28) | 2.50 (0.68) | 45.81 (2.25) | 7.05 (2.12) | 5.51 (3.95) |
| Recidivism Dataset | MLP t-SNE* | 89.22 (4.72) | 78.96 (6.90) | 5.58 (1.96) | 44.49 (3.47) | 6.00 (4.26) | 6.39 (5.55) |

