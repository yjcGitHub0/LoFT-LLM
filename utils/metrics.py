import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    mask = true > 1e-3
    if np.sum(mask) == 0: return 0
    return np.mean(np.abs((true[mask] - pred[mask]) / true[mask]))

def MAPE2(pred, true):
    mask = true > 1e-3
    if np.sum(mask) == 0: return 0
    return np.mean(np.abs((true[mask] - pred[mask]) / true[mask]))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def SMAPE(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    denominator = (np.abs(true) + np.abs(pred)) / 2
    # 为避免除以0，加个很小的eps
    eps = 1e-8
    smape_values = np.abs(pred - true) / (denominator + eps)
    return np.mean(smape_values)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)

    return mae, mse, rmse, mape, mspe, smape


def metric_nor(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE2(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)

    return mae, mse, rmse, mape, mspe, smape