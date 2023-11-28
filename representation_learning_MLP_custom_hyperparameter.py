# Zulfidin Khodzhaev

import numpy as np
import pickle as pkl
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

def fair_metric(pred, labels, sens):
    
    idx_s0 = sens == 0
    idx_s1 = sens == 1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels == 1)
    parity = abs(sum(pred[idx_s0]) / sum(idx_s0) - sum(pred[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred[idx_s1_y1]) / sum(idx_s1_y1))
    
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

    hidden_layer_sizes_space = [(30,), (50,), (100,), (50, 50), (30, 30), (100, 100), (100, 30), (30, 100), (100, 50), (50, 100)]
    activation_space = ['identity', 'logistic', 'tanh', 'relu']
    learning_rate_space = [0.001, 0.01, 0.1]
    batch_size_space = [32, 64, 128]
    alpha_space = [0.0001, 0.001, 0.01]
    best_accuracy = 0

    if extraction_option == 'PCA':
        
        for n_components in range(1, len(X_train_original[0]) + 1):
            pca = PCA(n_components=n_components, random_state=random_state)
            X_train = pca.fit_transform(X_train_original)
            X_test = pca.transform(X_test_original)
            for hidden_layer_sizes in hidden_layer_sizes_space:
                for activation in activation_space:
                    for learning_rate in learning_rate_space:
                        for batch_size in batch_size_space:
                            for alpha in alpha_space:
                                mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, learning_rate_init=learning_rate, batch_size=batch_size, alpha=alpha, random_state=random_state)
                                mlp.fit(X_train, y_train)
                                score = mlp.score(X_test, y_test)
                                if score > best_accuracy:
                                    best_hidden_layer_sizes = hidden_layer_sizes
                                    best_activation = activation
                                    best_learning_rate = learning_rate
                                    best_batch_size = batch_size
                                    best_alpha = alpha
                                    best_accuracy = score
                                    best_n_components = n_components
        print(f"PCA Best Config: N_components = {best_n_components}, Hidden Layer Sizes: {best_hidden_layer_sizes}, Activation: {best_activation}, Learning Rate: {best_learning_rate}, Batch Size: {best_batch_size}, Alpha: {best_alpha}, Test Accuracy: {best_accuracy}")
        return best_hidden_layer_sizes, best_activation, best_learning_rate, best_batch_size, best_alpha, best_n_components
    else:
        for hidden_layer_sizes in hidden_layer_sizes_space:
            for activation in activation_space:
                for learning_rate in learning_rate_space:
                    for batch_size in batch_size_space:
                        for alpha in alpha_space:
                            mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, learning_rate_init=learning_rate, batch_size=batch_size, alpha=alpha, random_state=random_state)
                            mlp.fit(X_train, y_train)
                            score = mlp.score(X_test, y_test)
                            if score > best_accuracy:
                                best_hidden_layer_sizes = hidden_layer_sizes
                                best_activation = activation
                                best_learning_rate = learning_rate
                                best_batch_size = batch_size
                                best_alpha = alpha
                                best_accuracy = score
        print(f"No Feature Extraction Best Config: Hidden Layer Sizes: {best_hidden_layer_sizes}, Activation: {best_activation}, Learning Rate: {best_learning_rate}, Batch Size: {best_batch_size}, Alpha: {best_alpha}, Test Accuracy: {best_accuracy}")
        
        return best_hidden_layer_sizes, best_activation, best_learning_rate, best_batch_size, best_alpha, None

    
def mlp_main(pkl_file, extraction_option, random_state):
    
    print(f'Dataset: {pkl_file}, Extraction Option: {extraction_option}, Random Seed: {random_state}')
    data = pkl.load(open(pkl_file, 'rb'))
    sens_idx = pkl_file_to_sens_idx[pkl_file]
    y_train = data['Y'][data['idx_train']]
    y_test = data['Y'][data['idx_test']]
    best_params = regression(data, y_train, y_test, extraction_option, random_state)
    clf = MLPClassifier(hidden_layer_sizes=best_params[0], activation=best_params[1], learning_rate_init=best_params[2], batch_size=best_params[3], alpha=best_params[4], random_state=random_state)
    clf.fit(data['X'][data['idx_train']], y_train)
    output = clf.predict(data['X'][data['idx_test']])
    auc_roc_test = roc_auc_score(y_test, clf.predict_proba(data['X'][data['idx_test']])[:, 1])
    parity, equality = fair_metric(output, y_test, data['sensitive_attr'][data['idx_test']])
    f1_s = f1_score(y_test, output)

    return auc_roc_test, f1_s, parity, equality

for pkl_file in pkl_file_to_sens_idx.keys():
    
    for extraction_option in ['PCA', 't-SNE', None]:
        
        auroc_list = []
        f1_score_list = []
        parity_list = []
        equality_list = []
        for random_seed in range(1, 6):
            np.random.seed(random_seed)
            random.seed(random_seed)
            auroc, f1, parity, equality = mlp_main(pkl_file, extraction_option, random_seed)
            auroc_list.append(auroc * 100)
            f1_score_list.append(f1 * 100)
            parity_list.append(parity * 100)
            equality_list.append(equality * 100)
 
        print(f'Dataset: {pkl_file}, Extraction Option: {extraction_option}')
        print(f'AUROC Mean: {np.average(auroc_list)}, STD: {np.std(auroc_list)}')
        print(f'F-1 Score Mean: {np.average(f1_score_list)}, STD: {np.std(f1_score_list)}')
        print(f'Parity Mean: {np.average(parity_list)}, STD: {np.std(parity_list)}')
        print(f'Equality Mean: {np.average(equality_list)}, STD: {np.std(equality_list)}')