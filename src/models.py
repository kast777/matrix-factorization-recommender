import numpy as np
import time

def train_nmf_manual(train_data, n_users, n_items, r, lr, reg, epochs, batch_size):
    """
    NMF manual com SGD e restricao de nao-negatividade.

    Argumentos:
        train_data (np.array): Matriz de interacoes [user_idx, item_idx, rating].
        n_users, n_items (int): Dimensoes da matriz original.
        r (int): Rank da matriz (dimensao latente).
        lr (float): Taxa de aprendizagem.
        reg (float): Parametro de regularizacao L2.
        epochs (int): Numero de passagens pelo dataset.
        batch_size (int): Tamanho do lote para o SGD.
    """
    U = np.random.rand(n_users, r) * 0.1
    V = np.random.rand(n_items, r) * 0.1
    num_samples = train_data.shape[0]
    
    for epoch in range(epochs):
        start_time = time.time()
        np.random.shuffle(train_data)
        total_loss = 0
        for i in range(0, num_samples, batch_size):
            batch = train_data[i:i+batch_size]
            u_idxs, i_idxs, ratings = batch[:, 0].astype(int), batch[:, 1].astype(int), batch[:, 2].astype(float)
            
            preds = np.sum(U[u_idxs] * V[i_idxs], axis=1)
            error = ratings - preds
            total_loss += np.sum(error**2)
            
            U[u_idxs] += lr * (error[:, None] * V[i_idxs] - reg * U[u_idxs])
            V[i_idxs] += lr * (error[:, None] * U[u_idxs] - reg * V[i_idxs])
            
            U[u_idxs] = np.maximum(U[u_idxs], 0)
            V[i_idxs] = np.maximum(V[i_idxs], 0)
        
        print(f"Epoch {epoch+1}/{epochs} NMF | Loss: {total_loss:.1f} | {time.time()-start_time:.2f}s")
    return U, V

def train_wmf_manual(train_data, num_users, num_items, r, k_neg, alpha, lr, reg, epochs, batch_size):
    """
    WMF manual com Negative Sampling e pesos de confianca diferenciados.

    Argumentos:
        train_data (np.array): Matriz de interacoes positivas [user_idx, item_idx, 1].
        num_users (int): Total de utilizadores únicos.
        num_items (int): Total de itens (filmes) únicos.
        r (int): Numero de fatores latentes (rank).
        k_neg (int): Ratio de amostras negativas (quantos 'zeros' por cada 'um').
        alpha (float): Coeficiente de confianca (C_ui = 1 + alpha * R_ui).
        lr (float): Learning rate para o SGD.
        reg (float): Parametro de regularizacao L2 (lambda).
        epochs (int): Numero de iteracoes sobre o dataset.
        batch_size (int): Tamanho do mini-batch.
    """
    U = np.random.normal(0, 0.01, (num_users, r))
    V = np.random.normal(0, 0.01, (num_items, r))
    num_samples = train_data.shape[0]
    
    for epoch in range(epochs):
        start_time = time.time()
        np.random.shuffle(train_data)
        total_loss = 0
        for i in range(0, num_samples, batch_size):
            batch_pos = train_data[i:i+batch_size]
            u_pos, i_pos = batch_pos[:, 0].astype(int), batch_pos[:, 1].astype(int)
            
            # Positivos
            C_pos = 1.0 + alpha
            pred_pos = np.sum(U[u_pos] * V[i_pos], axis=1)
            err_pos = 1.0 - pred_pos
            U[u_pos] += lr * (C_pos * (err_pos[:, None] * V[i_pos]) - reg * U[u_pos])
            V[i_pos] += lr * (C_pos * (err_pos[:, None] * U[u_pos]) - reg * V[i_pos])
            total_loss += np.sum(C_pos * err_pos**2)
            
            # Negativos (Negative Sampling)
            u_neg = np.repeat(u_pos, k_neg)
            i_neg = np.random.randint(0, num_items, size=(len(u_pos) * k_neg))
            pred_neg = np.sum(U[u_neg] * V[i_neg], axis=1)
            err_neg = 0.0 - pred_neg
            
            np.add.at(U, u_neg, lr * (err_neg[:, None] * V[i_neg] - reg * U[u_neg]))
            np.add.at(V, i_neg, lr * (err_neg[:, None] * U[u_neg] - reg * V[i_neg]))
            total_loss += np.sum(err_neg**2)

        print(f"Epoch {epoch+1}/{epochs} WMF | Loss: {total_loss:.1f} | {time.time()-start_time:.2f}s")
    return U, V
