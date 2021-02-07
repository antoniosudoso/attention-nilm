import numpy as np


def mae(prediction, true):
    MAE = abs(true - prediction)
    MAE = np.sum(MAE)
    MAE = MAE / len(prediction)
    return MAE


def sae(prediction, true, N):
    T = len(prediction)
    K = int(T / N)
    SAE = 0
    for k in range(1, N):
        pred_r = np.sum(prediction[k * N: (k + 1) * N])
        true_r = np.sum(true[k * N: (k + 1) * N])
        SAE += abs(true_r - pred_r)
    SAE = SAE / (K * N)
    return SAE


def f1(prediction, true):
    epsilon = 1e-8
    TP = epsilon
    FN = epsilon
    FP = epsilon
    TN = epsilon
    for i in range(len(prediction)):
        if prediction[i] >= 0.5:
            prediction_binary = 1
        else:
            prediction_binary = 0
        if prediction_binary == 1 and true[i] == 1:
            TP += 1
        elif prediction_binary == 0 and true[i] == 1:
            FN += 1
        elif prediction_binary == 1 and true[i] == 0:
            FP += 1
        elif prediction_binary == 0 and true[i] == 0:
            TN += 1
    R = TP / (TP + FN)
    P = TP / (TP + FP)
    f1 = (2 * P * R) / (P + R)
    return f1


def standardize_data(data, mu=0.0, sigma=1.0):
    data -= mu
    data /= sigma
    return data


def normalize_data(data, min_value=0.0, max_value=1.0):
    data -= min_value
    data /= max_value - min_value
    return data


def build_overall_sequence(sequences):
    unique_sequence = []
    matrix = [sequences[::-1, :].diagonal(i) for i in range(-sequences.shape[0] + 1, sequences.shape[1])]
    for i in range(len(matrix)):
        unique_sequence.append(np.median(matrix[i]))
    unique_sequence = np.array(unique_sequence)
    return unique_sequence
