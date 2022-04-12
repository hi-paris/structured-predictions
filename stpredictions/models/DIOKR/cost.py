import torch


def sloss(Omega, K_x_tr_ba, K_y_tr_ba, K_y_ba_ba, K_y):
    n_ba = K_x_tr_ba.shape[1]

    mse = (1.0 / n_ba) * torch.trace(K_x_tr_ba.T @ Omega @ K_y @ Omega.T @ K_x_tr_ba
                                     - 2 * K_x_tr_ba.T @ Omega @ K_y_tr_ba + K_y_ba_ba)

    return mse


def sloss_batch(Omega_block_diag, K_x_tr_te, K_y_tr_te, K_y_te_te, K_y, n_b):
    n_te = K_x_tr_te.shape[1]

    se = torch.diag((1.0 / n_b ** 2) * K_x_tr_te.T @ Omega_block_diag @ K_y @ Omega_block_diag.T @ K_x_tr_te
                    - (2.0 / n_b) * K_x_tr_te.T @ Omega_block_diag @ K_y_tr_te + K_y_te_te)

    mse = torch.mean(se)
    std = torch.std(se) / n_te ** (1 / 2)

    return mse, std
