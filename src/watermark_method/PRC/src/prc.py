import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.special import binom
from ldpc import bp_decoder
import sys
import galois

GF = galois.GF(2)

def boolean_row_reduce(A, print_progress=False):
    n, k = A.shape
    A_rr = A.copy()
    perm = np.arange(n)
    for j in range(k):
        idxs = j + np.nonzero(A_rr[j:, j])[0]
        if idxs.size == 0:
            return None
        A_rr[[j, idxs[0]]] = A_rr[[idxs[0], j]]
        (perm[j], perm[idxs[0]]) = (perm[idxs[0]], perm[j])
        A_rr[idxs[1:]] += A_rr[j]
    return perm[:k]

def KeyGen(n, message_length=512, false_positive_rate=1e-6, t=3, g=None, r=None, noise_rate=None):
    num_test_bits = int(np.ceil(np.log2(1 / false_positive_rate)))
    secpar = int(np.log2(binom(n, t)))
    if g is None: g = secpar
    if noise_rate is None: noise_rate = 0.01 
    k = message_length + g + num_test_bits
    if r is None: r = n - k - secpar

    generator_matrix = GF.Random((n, k))
    row_indices, col_indices, data = [], [], []
    for row in range(r):
        chosen_indices = np.random.choice(n - r + row, t - 1, replace=False)
        chosen_indices = np.append(chosen_indices, n - r + row)
        row_indices.extend([row] * t)
        col_indices.extend(chosen_indices)
        data.extend([1] * t)
        generator_matrix[n - r + row] = generator_matrix[chosen_indices[:-1]].sum(axis=0)
    
    parity_check_matrix = csr_matrix((data, (row_indices, col_indices)))
    max_bp_iter = int(np.log(n) / np.log(t))
    one_time_pad = GF.Random(n)
    test_bits = GF.Random(num_test_bits)

    permutation = np.random.permutation(n)
    generator_matrix = generator_matrix[permutation]
    one_time_pad = one_time_pad[permutation]
    parity_check_matrix = parity_check_matrix[:, permutation]

    encoding_key = (generator_matrix, one_time_pad, test_bits, g, noise_rate)
    decoding_key = (generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate, noise_rate, test_bits, g, max_bp_iter, t)
    return encoding_key, decoding_key

def Encode(encoding_key, message=None):
    generator_matrix, one_time_pad, test_bits, g, noise_rate = encoding_key
    n, k = generator_matrix.shape

    if message is None:
        msg_bits = GF.Random(k - len(test_bits) - g)
    else:
        msg_bits = GF(message.detach().cpu().numpy().flatten().astype(np.int64)) if torch.is_tensor(message) else GF(np.array(message).flatten().astype(np.int64))
            
    padding_len = k - len(test_bits) - g - len(msg_bits)
    payload = np.concatenate((test_bits, GF.Random(g), msg_bits, GF.Zeros(padding_len)))
    
    clean_codeword = payload @ generator_matrix.T + one_time_pad
    
    error = GF(np.random.binomial(1, noise_rate, n))
    final_codeword = clean_codeword + error
    
    return 1 - 2 * torch.tensor(np.array(final_codeword, dtype=float)), np.array(clean_codeword, dtype=np.int64), np.array(msg_bits, dtype=np.int64)

def Detect(decoding_key, posteriors, false_positive_rate=None):
    generator_matrix, parity_check_matrix, one_time_pad, fpr_key, noise_rate, _, _, _, t = decoding_key
    fpr = false_positive_rate if false_positive_rate is not None else fpr_key
    posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)
    r = parity_check_matrix.shape[0]
    Pi = np.prod(posteriors[parity_check_matrix.indices.reshape(r, t)], axis=1)
    log_plus = np.log((1 + Pi + 1e-12) / 2)
    log_minus = np.log((1 - Pi + 1e-12) / 2)
    log_prod = log_plus + log_minus
    threshold = np.sqrt(2 * np.sum(np.power(log_plus, 2) + np.power(log_minus, 2) - 0.5 * np.power(log_prod, 2)) * np.log(1 / fpr)) + 0.5 * log_prod.sum()
    return log_plus.sum() >= threshold

def Decode(decoding_key, posteriors, max_bp_iter=None):
    generator_matrix, parity_check_matrix, one_time_pad, _, noise_rate, test_bits, g, max_bp_iter_key, t = decoding_key
    max_iter = max_bp_iter if max_bp_iter is not None else max_bp_iter_key

    posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)
    channel_probs = (1 - np.abs(posteriors)) / 2
    x_recovered = (1 - np.sign(posteriors)) // 2

    bpd = bp_decoder(parity_check_matrix, channel_probs=channel_probs, max_iter=max_iter, bp_method="product_sum")
    x_decoded = bpd.decode(x_recovered.astype(np.int64))

    bpd_probs = 1 / (1 + np.exp(np.clip(bpd.log_prob_ratios, -20, 20)))
    confidences = 2 * np.abs(0.5 - bpd_probs)
    confidence_order = np.argsort(-confidences)

    top_invertible_rows = boolean_row_reduce(generator_matrix[confidence_order])
    if top_invertible_rows is None: return None

    try:
        recovered_payload = np.linalg.solve(generator_matrix[confidence_order][top_invertible_rows], GF(x_decoded[confidence_order][top_invertible_rows]))
        return np.array(recovered_payload, dtype=np.int64)
    except:
        return None