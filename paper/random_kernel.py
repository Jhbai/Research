import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy

class RandomKernelFeatureExtractor:
    def __init__(self, num_kernels=1000):
        self.num_kernels = num_kernels
        self.kernels = []
        self.biases = []

    def _generate_weights(self):
        weights = np.full(9, -1)
        indices = np.random.choice(9, 3, replace=False)
        weights[indices] = 2
        return weights

    def fit_transform(self, X_subseqs):
        N, L = X_subseqs.shape
        max_exponent = math.log2((L - 1) / 8) if L > 9 else 0
        dilations = [int(2**i) for i in np.linspace(0, max_exponent, int(max_exponent)+1)]
        
        features = np.zeros((N, self.num_kernels))
        self.kernels = []
        
        for i in range(self.num_kernels):
            w = self._generate_weights()
            d = np.random.choice(dilations) if dilations else 1
            p = np.random.choice([True, False])
            
            kernel_length = (9 - 1) * d + 1
            
            conv_outputs = []
            for j in range(N):
                subseq = X_subseqs[j]
                if p:
                    pad_len = kernel_length // 2
                    subseq = np.pad(subseq, (pad_len, pad_len), 'constant', constant_values=0)
                
                if len(subseq) < kernel_length:
                    out = np.zeros(1)
                else:
                    out_len = len(subseq) - kernel_length + 1
                    out = np.zeros(out_len)
                    for t in range(out_len):
                        out[t] = np.sum(subseq[t : t + 9*d : d] * w)
                conv_outputs.append(out)
            
            random_idx = np.random.randint(N)
            if len(conv_outputs[random_idx]) > 0:
                bias = np.quantile(conv_outputs[random_idx], np.random.uniform(0, 1))
            else:
                bias = 0
                
            self.kernels.append({'w': w, 'd': d, 'p': p, 'bias': bias})
            
            for j in range(N):
                if len(conv_outputs[j]) > 0:
                    ppv = np.mean((conv_outputs[j] - bias) > 0)
                else:
                    ppv = 0
                features[j, i] = ppv
                
        return features

    def transform(self, X_subseqs):
        N, L = X_subseqs.shape
        features = np.zeros((N, self.num_kernels))
        
        for i, k_params in enumerate(self.kernels):
            w = k_params['w']
            d = k_params['d']
            p = k_params['p']
            bias = k_params['bias']
            kernel_length = (9 - 1) * d + 1
            
            for j in range(N):
                subseq = X_subseqs[j]
                if p:
                    pad_len = kernel_length // 2
                    subseq = np.pad(subseq, (pad_len, pad_len), 'constant', constant_values=0)
                
                if len(subseq) < kernel_length:
                    features[j, i] = 0
                    continue
                    
                out_len = len(subseq) - kernel_length + 1
                out = np.zeros(out_len)
                for t in range(out_len):
                    out[t] = np.sum(subseq[t : t + 9*d : d] * w)
                    
                features[j, i] = np.mean((out - bias) > 0)
        return features


class KernelSelector:
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, bins=10):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bins = bins
        self.selected_indices = None

    def fit_transform(self, features):
        N, M = features.shape
        discrete_features = np.zeros_like(features, dtype=int)
        for i in range(M):
            hist, bin_edges = np.histogram(features[:, i], bins=self.bins)
            discrete_features[:, i] = np.digitize(features[:, i], bin_edges[:-1])
        
        kss_scores = np.zeros(M)
        for i in range(M):
            f_i = discrete_features[:, i]
            _, counts_i = np.unique(f_i, return_counts=True)
            p_i = counts_i / N
            H_i = entropy(p_i)
            
            MI_sum = 0
            for j in range(M):
                if i != j:
                    f_j = discrete_features[:, j]
                    MI_sum += mutual_info_score(f_i, f_j)
            
            avg_MI = MI_sum / (M - 1) if M > 1 else 0
            kss_scores[i] = self.alpha * avg_MI - self.beta * H_i
            
        num_select = max(1, int(M * self.gamma))
        self.selected_indices = np.argsort(kss_scores)[::-1][:num_select]
        return features[:, self.selected_indices]

    def transform(self, features):
        return features[:, self.selected_indices]
