import numpy as np
import os
import csv
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import scipy.spatial
import scipy.linalg
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.optimize import check_grad
from functools import partial
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats
from scipy.stats import zscore


datas = np.load("predictions_saved.npy", allow_pickle=True).item()

def extract_point_clouds(pred_dict):

    X_pred, Y_pred = [], []  # Vettori delle coordinate predette
    X_real, Y_real = [], []  # Vettori delle coordinate reali
    img_names = []  # Lista dei nomi delle immagini

    # Iteriamo su tutte le immagini nel dizionario
    for img_name, joints in pred_dict.items():
        joints = np.array(joints)  # Convertiamo in numpy array
    
        x_pred_joints = joints[:, 0, 0]  # Tutte le X predette
        y_pred_joints = joints[:, 0, 1]  # Tutte le Y predette
        x_real_joints = joints[:, 1, 0]  # Tutte le X reali
        y_real_joints = joints[:, 1, 1]  # Tutte le Y reali

        # Aggiungiamo i dati alle liste
        X_pred.append(x_pred_joints)
        Y_pred.append(y_pred_joints)
        X_real.append(x_real_joints)
        Y_real.append(y_real_joints)
        img_names.append(img_name)  # Salviamo anche il nome dell'immagine

    # Convertiamo tutto in NumPy array per efficienza
    return np.array(X_pred), np.array(Y_pred), np.array(X_real), np.array(Y_real), img_names

X_pred,Y_pred,X_real,Y_real,img_names=extract_point_clouds(datas)

X_pred = X_pred.reshape(-1, 1)  
Y_pred = Y_pred.reshape(-1, 1)

X= np.hstack((X_pred, Y_pred))  #nuvola di partenza 

X_real = X_real.reshape(-1, 1)  # Shape (N, 1)
Y_real=  Y_real.reshape(-1, 1)  # Shape (N, 1)

Y = np.hstack((X_real, Y_real))

def remove_2d_outliers(points1,points2, soglia):
   
    zscores = np.abs(zscore(points1, axis=0))
    mask = (zscores < soglia).all(axis=1)
    return points1[mask],points2[mask]

Y,X=remove_2d_outliers(Y,X,3)


def rototraslazione(angolo,X):
    
    sigma=5

    angolo=np.radians(angolo)

    matrice_rotazione = np.array([[np.cos(angolo), -np.sin(angolo)],[np.sin(angolo),  np.cos(angolo)]])

    traslazione = np.array([5, 20])

    P_trasformata = np.dot(X, matrice_rotazione) + traslazione 

    rumore=np.random.normal(4,sigma,X.shape)

    P_trasformata+=rumore

    return P_trasformata

#Y=rototraslazione(65,X)

K=50

def RBF_esponenziale(C1,C2,X,sigma):
    X1, X2 = X[:, 0][:, np.newaxis], X[:, 1][:, np.newaxis]
    matrix = (X1 - C1)**2  + (X2 - C2)**2
    return  np.exp(-matrix/2*sigma)

def RBF_logaritmica(C1,C2,X,sigma):
    X1, X2 = X[:, 0][:, np.newaxis], X[:, 1][:, np.newaxis]
    matrix = (X1 - C1)**2  + (X2 - C2)**2
    return  np.log(1+sigma*matrix)

def RBF_multiquadratica(C1,C2,X,sigma):
    X1, X2 = X[:, 0][:, np.newaxis], X[:, 1][:, np.newaxis]
    matrix = (X1 - C1)**2  + (X2 - C2)**2
    return  np.sqrt(matrix+sigma**2)

def matrix_A(C,X,sigma,epsilon,func):
    C1=C[:,0]
    C2=C[:,1]
    starting_matrix=func(C1,C2,X,sigma)
    N, K = starting_matrix.shape
    zero_block = np.zeros((N, K), dtype= starting_matrix.dtype)
    top = np.hstack((starting_matrix, zero_block)) 
    bottom = np.hstack((zero_block, starting_matrix)) 
    matrix= np.vstack((top, bottom)) 
    I = np.identity(2*K)*(np.sqrt(epsilon))
    A = np.vstack((matrix, I))
    return A 

def vector_b(Y,K):
    vettore=Y.flatten(order='F')
    zeri=np.zeros(2*K)
    b = np.concatenate((vettore, zeri))
    return b 

def gradiente_coniugato(W,C,X,Y,sigma,epsilon,tol,func,K):
    k=0
    A=matrix_A(C,X,sigma,epsilon,func)
    r=vector_b(Y,K)-np.dot(A,W) #(2N+2K)
    d=np.dot(A.T,r) #2K
    #print('la prima direzione è',d)
    gamma=np.linalg.norm(np.dot(A.T,r))**2 
    #print('la norma del gradiente è', gamma)
    #print('questo è W prima di essere ottimizzato',W)
    while gamma>=tol:
        q=np.dot(A,d) #2n+2k
        alfa=gamma/(np.linalg.norm(q)**2) #scalare
        W=W+alfa*d #2k
        #print('ho aggiornato W',W)
        r=r-alfa*q #2N+2K
        g=-np.dot(A.T,r)
        gamma_nuovo=np.linalg.norm(g)**2
        #print('la nuova norma del gradiente è',gamma_nuovo)
        beta=gamma_nuovo/gamma
        gamma=gamma_nuovo
        d=-g+beta*d
        #print('la nuova direzione è',d)
        k+=1
    print('norma gradiente',gamma)
    diff=np.dot(A,W)-vector_b(Y,K)
    norma=0.5*((np.linalg.norm(diff))**2)
    return W,norma

def plot_selected_points(clouds, K=16):
   
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    labels = ['Nuvola 1', 'Nuvola 2', 'Nuvola 3']

    plt.figure(figsize=(8, 6))

    for i, cloud in enumerate(clouds):
        plt.scatter(cloud[:K, 0], cloud[:K, 1], color=colors[i], marker=markers[i], label=labels[i], alpha=0.7)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(f'Visualizzazione dei primi {K} punti di 3 nuvole di punti')
    plt.grid(True)
    plt.show()


def nuvola_predetta(W,X,C,K,sigma,func):
    C1=C[:,0]
    C2=C[:,1]
    matrice=func(C1,C2,X,sigma)
    W_matrix=np.column_stack((W[:K], W[K:]))
    cloud3=np.dot(matrice,W_matrix)
    return cloud3

def compute_k_centers(points, k):
    kmeans = KMeans(n_clusters=k, random_state=32)
    kmeans.fit(points)
    return kmeans.cluster_centers_, kmeans.labels_

centers, labels = compute_k_centers(X,K)
sigmus=[10,20,30,40,50,60,80,100,200,300,400,500,600]
rbf_functions =[RBF_multiquadratica]
diz={}
for RBF in rbf_functions:
    nome_RBF = RBF.__name__  # Ottiene il nome della funzione come stringa
    for sigma in sigmus:
        #print(f'Sto processando RBF={nome_RBF}, sigma={sigma}')
        lista = []
        np.random.seed(32)
        W = np.random.random(2*K)
        C = centers
        epsilon = 0.5
        tol = 10**-6
        # Calcolo con gradiente coniugato
        W, norma= gradiente_coniugato(W, C, X, Y, sigma, epsilon, tol, RBF,K)
        # Salviamo il valore della norma
        lista.append(norma)
        # Salviamo i risultati nel dizionario
        diz[(RBF, sigma)] = lista
# Troviamo il valore minimo
min_key, min_value = min(diz.items(), key=lambda x: min(x[1]))

# Stampiamo il miglior risultato
print(f"Il miglior risultato è stato ottenuto con:")
print(f"- RBF: {min_key[0]}")
print(f"- Sigma: {min_key[1]}")
print(f"- Valore minimo della norma: {min_value}")
best_rbf = min_key[0]
best_sigma = min_key[1]
W=np.random.random(2*K)
C=centers
epsilon=0.5
tol=10**-10
W_opt,norma=gradiente_coniugato(W,C,X,Y,best_sigma,epsilon,tol,best_rbf,K)
print('norma',norma)
cloud3=nuvola_predetta(W_opt,X,C,K,best_sigma,best_rbf)
clouds=[X,Y,cloud3]
plot_selected_points(clouds,10)




























    
