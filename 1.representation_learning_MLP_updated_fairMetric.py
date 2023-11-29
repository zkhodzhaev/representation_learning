import numpy as np
import pickle as pkl
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# from google.colab import files
# uploaded = files.upload()

def fair_metric(pred1, pred2, labels, sens):
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred1[idx_s0]) / sum(idx_s0) - sum(pred2[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred1[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred2[idx_s1_y1]) / sum(idx_s1_y1))
    return parity.item(), equality.item()

pkl_file_to_sens_idx = {
    'german_dset.pkl': 0,
    'credit_dset.pkl': 1,
    'bail_dset.pkl': 0
}

def regression(data, y_train, y_test, extraction_option, random_state):
    X_train = data['X'][data['idx_train']]
    X_test = data['X'][data['idx_test']]

    X_train_original = X_train.copy()
    X_test_original = X_test.copy()

    hidden_layer_sizes_space = [(50,), (100,), (50, 50), (100, 100), (100, 50), (50, 100)]
    activation_space = ['identity', 'logistic', 'tanh', 'relu']
    best_accuracy = 0

    if extraction_option == 'PCA':
        for n_components in range(1, len(X_train_original[0]) + 1):
            pca = PCA(n_components=n_components, random_state=random_state)
            pca = pca.fit(X_train_original)
            X_train = pca.transform(X_train_original)
            X_test = pca.transform(X_test_original)

            for hidden_layer_sizes in hidden_layer_sizes_space:
                for activation in activation_space:
                    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, random_state=random_state)
                    mlp.fit(X_train, y_train)
                    score = mlp.score(X_test, y_test)
                    if score > best_accuracy:
                        best_hidden_layer_sizes = hidden_layer_sizes
                        best_activation = activation
                        best_n_components = n_components
                        best_accuracy = score
        print(f"PCA N_components = {best_n_components}")
        print(f"MLP Hidden Layer Sizes: {best_hidden_layer_sizes}, Activation: {best_activation}, Test Accuracy: {best_accuracy}")
        return best_hidden_layer_sizes, best_activation, best_n_components

    elif extraction_option == 't-SNE':
        for n_components in range(1, 4):  # Restrict to a maximum of 3 components for t-SNE
            tsne = TSNE(n_components=n_components, random_state=random_state, method='barnes_hut')
            X = tsne.fit_transform(data['X'])
            X_train = X[data['idx_train']]
            X_test = X[data['idx_test']]

            for hidden_layer_sizes in hidden_layer_sizes_space:
                for activation in activation_space:
                    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, random_state=random_state)
                    mlp.fit(X_train, y_train)
                    score = mlp.score(X_test, y_test)
                    if score > best_accuracy:
                        best_hidden_layer_sizes = hidden_layer_sizes
                        best_activation = activation
                        best_n_components = n_components
                        best_accuracy = score
        print(f"t-SNE N_components = {best_n_components}")
        print(f"MLP Hidden Layer Sizes: {best_hidden_layer_sizes}, Activation: {best_activation}, Test Accuracy: {best_accuracy}")
        return best_hidden_layer_sizes, best_activation, best_n_components

    else:
        for hidden_layer_sizes in hidden_layer_sizes_space:
            for activation in activation_space:
                mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, random_state=random_state)
                mlp.fit(X_train, y_train)
                score = mlp.score(X_test, y_test)
                if score > best_accuracy:
                    best_hidden_layer_sizes = hidden_layer_sizes
                    best_activation = activation
                    best_accuracy = score
        print(f"MLP Hidden Layer Sizes: {best_hidden_layer_sizes}, Activation: {best_activation}, Test Accuracy: {best_accuracy}")
        return best_hidden_layer_sizes, best_activation, None
    
    
def counterfactual_fairness(original_output, counter_output):
    return 1 - accuracy_score(original_output, counter_output)

def robustness_score(original_output, noisy_output):
    return 1 - accuracy_score(original_output, noisy_output)

def augment_data(X, sens_idx):
    """
    Randomly flip the sensitive attribute to augment the data.
    """
    augmented_X = X.copy()
    flip_indices = np.random.choice([True, False], size=len(X))
    augmented_X[flip_indices, sens_idx] = 1 - augmented_X[flip_indices, sens_idx]
    return augmented_X


def mlp_main(pkl_file, extraction_option, random_state, augment=False):
    print(f'Dataset: {pkl_file}, Extraction Option: {extraction_option}, Random Seed: {random_state}, Data Augmentation: {augment}')
    data = pkl.load(open(pkl_file, 'rb'))
    sens_idx = pkl_file_to_sens_idx[pkl_file]

    X_train = data['X'][data['idx_train']]
    y_train = data['Y'][data['idx_train']]
    X_test = data['X'][data['idx_test']]
    y_test = data['Y'][data['idx_test']]

    if augment:
        X_train = augment_data(X_train, sens_idx)

    best_hidden_layer_sizes, best_activation, best_n_components = regression(data, y_train, y_test, extraction_option, random_state)

    pca, tsne = None, None
    if extraction_option == 'PCA':
        pca = PCA(n_components=best_n_components, random_state=random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
    elif extraction_option == 't-SNE':
        tsne = TSNE(n_components=best_n_components, random_state=random_state, method='barnes_hut')
        combined_X = np.vstack((X_train, X_test))
        combined_X_transformed = tsne.fit_transform(combined_X)
        X_train = combined_X_transformed[:len(X_train)]
        X_test = combined_X_transformed[len(X_train):]


    clf = MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes, activation=best_activation, random_state=random_state).fit(X_train, y_train)

    Features1, Features2 = data['X'].copy(), data['X'].copy()
    Features1[:, sens_idx], Features2[:, sens_idx] = 0, 1

    if pca:
        Features1 = pca.fit_transform(Features1)
        Features2 = pca.fit_transform(Features2)
    elif tsne:
        Features1 = tsne.fit_transform(Features1)
        Features2 = tsne.fit_transform(Features2)

    pred1 = clf.predict(Features1[data['idx_test']])
    pred2 = clf.predict(Features2[data['idx_test']])
    parity, equality = fair_metric(pred1, pred2, y_test, data['sensitive_attr'][data['idx_test']])

    output = clf.predict(X_test)
    auc_roc_test = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    f1_s = f1_score(y_test, output)

    counter_features = data['X'].copy()
    counter_features[:, sens_idx] = 1 - counter_features[:, sens_idx]
    noisy_features = np.add(data['X'].copy(), np.random.normal(0, 1, data['X'].shape))

    clf_counter = MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes, activation=best_activation, random_state=random_state).fit(counter_features[data['idx_train']], y_train)
    clf_noisy = MLPClassifier(hidden_layer_sizes=best_hidden_layer_sizes, activation=best_activation, random_state=random_state).fit(noisy_features[data['idx_train']], y_train)

    counter_output = clf_counter.predict(counter_features[data['idx_test']])
    noisy_output = clf_noisy.predict(noisy_features[data['idx_test']])
    counterfactual_fairness_metric = counterfactual_fairness(output, counter_output)
    robustness_score_metric = robustness_score(output, noisy_output)

    return auc_roc_test, f1_s, counterfactual_fairness_metric, robustness_score_metric, parity, equality


for pkl_file in pkl_file_to_sens_idx.keys():
    for extraction_option in ['PCA', 't-SNE', None]:
        for augment in [False, True]:
            auroc_list = []
            f1_score_list = []
            counterfactual_fairness_list = []
            robustness_score_list = []
            parity_list = []
            equality_list = []
            for random_seed in range(1, 6):
                np.random.seed(random_seed)
                random.seed(random_seed)
                results = mlp_main(pkl_file, extraction_option, random_seed, augment)  # Pass augment here
                auroc_list.append(results[0]*100)
                f1_score_list.append(results[1]*100)
                counterfactual_fairness_list.append(results[2]*100)
                robustness_score_list.append(results[3]*100)
                parity_list.append(results[4]*100)
                equality_list.append(results[5]*100)

            print(f'Dataset: {pkl_file}, Extraction Option: {extraction_option}, Data Augmentation: {augment}')
            print(f'AUROC Mean: {np.average(auroc_list)}, STD: {np.std(auroc_list)}')
            print(f'F-1 Score Mean: {np.average(f1_score_list)}, STD: {np.std(f1_score_list)}')
            print(f'Counterfactual Fairness Mean (Unfairness Mean): {np.average(counterfactual_fairness_list)}, STD: {np.std(counterfactual_fairness_list)}')
            print(f'Robustness Score Mean (Instability Mean): {np.average(robustness_score_list)}, STD: {np.std(robustness_score_list)}')
            print(f'Parity Mean (Delta PS): {np.average(parity_list)}, STD: {np.std(parity_list)}')
            print(f'Equality Mean (Delta EO): {np.average(equality_list)}, STD: {np.std(equality_list)}')
