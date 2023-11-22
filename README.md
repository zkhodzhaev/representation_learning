### Representation Learning using Multi-Layer Perceptrons (MLPs)

Multi-layer perceptrons (MLPs) were selected for their versatility in handling diverse data types and intricate relationships. A strategic choice was made to utilize 50 and 100 nodes in the hidden layers of the MLPs to balance model complexity and computational efficiency. This decision was informed by the feature sizes of the datasets: the German Credit Dataset with 27 features, the Credit Defaulter Dataset with 13 features, and the Bail Dataset with 18 features. These configurations were designed to effectively capture patterns in datasets varying in feature count and sample size, ranging from 1,000 to 30,000 entries. The objective was to avert overfitting, especially in datasets with fewer features, while assessing the impact of various network architectures on performance.

The MLPs were configured with diverse hidden layer sizes and activation functions, including 'identity', 'logistic', 'tanh', and 'relu'. This range, from simpler single-layered networks to more complex structures like (50, 50) and (100, 100) multi-layered networks, set the stage for empirical experimentation and adjustments based on performance. 

To confront the challenges of high-dimensional data, dimensionality reduction techniques such as PCA and t-SNE were implemented. PCA was used to transform the feature space into principal components, thereby reducing complexity while preserving critical information. In contrast, t-SNE was employed to retain the local structure of data in a reduced-dimensional space, facilitating the interpretation of complex datasets.

The introduction of additional hyperparameters like learning rate, batch size, and regularization parameters was intended to allow more refined tuning of neural network training. However, this added complexity also made it more difficult to maintain consistent performance across experiments. Specifically, changes to these parameters can substantially alter the training dynamics, leading neural networks to converge differently and develop different generalization capabilities. To reduce this source of variability, this study used the default library settings for the learning rate, batch size, and regularization parameters.

#### Results Table

Here's a summary of the MLP results as compared to other methods:

| Dataset | Method | AUROC (↑) | F1-Score (↑) | Unfairness (↓) | Instability (↓) | ΔSP (↓) | ΔEO (↓) |
|---------|--------|-----------|--------------|----------------|-----------------|---------|---------|
| German Credit Graph | MLP | 76.02 (1.01) | 80.28 (1.88) | 14.64 (14.00) | 28.64 (5.36) | 28.49 (3.21) | 17.65 (2.95) |
| Credit Defaulter Graph | MLP | 74.92 (0.12) | 83.50 (0.17) | 1.24 (1.39) | 34.10 (3.23) | 15.78 (0.53) | 13.38 (0.75) |
| Recidivism Graph | MLP | 94.42 (0.18) | 88.43 (0.27) | 0.45 (0.12) | 42.32 (2.42) | 2.42 (0.28) | 3.57 (0.58) |
