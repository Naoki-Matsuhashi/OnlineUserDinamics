import random
import numpy as np
np.set_printoptions(threshold=np.inf)#npによる大型行列を全て表示
from math import sqrt#平方根
from math import fabs#少数まで扱う絶対値
import math
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import statistics as sta
import scipy.linalg
from mpl_toolkits.mplot3d import axes3d
np.set_printoptions(precision=8)



#----------------------------関数定義---------------------------------

def graph_print(graph): #グラフ描画
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, alpha=0.5)
    plt.show()

def sinΩt(t): #sin(Ωt)
    sin_sqrt_val_t=np.sin(sqrt_val*t)
    sinΩt=np.diag(sin_sqrt_val_t)
    return sinΩt

def cosΩt(t): #cin(Ωt)
    cos_sqrt_val_t=np.cos(sqrt_val*t)
    cosΩt=np.diag(cos_sqrt_val_t)
    return cosΩt

def y_diff(t): #y^+(t)-y^-(t)
    y=P @ cosΩt(t) @ inv_P @ y_diff0 - 1j * P @ Ʊ @ sinΩt(t) @ inv_P @ sqrt_D @ y_sum0
    return y

def y_sum(t): #y^+(t)-y^-(t)
    y=sqrt_inv_D @ P @ cosΩt(t) @ inv_P @ sqrt_D @ y_sum0 - 1j * sqrt_inv_D @ P @ Ω @ sinΩt(t) @ inv_P @ y_diff0
    return y


#----------------------------初期条件---------------------------------

n=41 #ノード数
n_list=list(range(n))
t=50 #時間

G=nx.cycle_graph(n) #グラフ

y_sum0=np.zeros(n,complex) #y^+(0)
y_diff0=np.zeros(n,complex)
y_diff0[20]=1 #y^-(0)



#----------------------------行列導出---------------------------------


A = nx.to_numpy_array(G) #隣接行列
D = np.diag(np.sum(A,axis=1))  #次数行列
L = D-A  #ラプラシアン行列
val, P = scipy.linalg.eigh(L) #固有値，固有ベクトル
inv_P = np.linalg.inv(P) #固有ベクトル逆行列
#Λ=inv_P@L@P #Lの固有値からなる対角行列 #誤差のため不採用
tmp_Λ = val.copy()
sqrt_val = np.sqrt(tmp_Λ) #固有値の平方根，sin,cosの計算に必要
Λ = np.diag(tmp_Λ) #Lの固有値からなる対角行列
Ω = scipy.linalg.sqrtm(Λ) #Λの平方根行列
Ω[0][0]=0
Ʊ=np.linalg.pinv(Ω) #Ωの擬似逆行列
sqrt_D = scipy.linalg.sqrtm(D) #次数行列の平方根行列
inv_D=np.linalg.inv(D) #次数行列の逆行列
sqrt_inv_D=scipy.linalg.sqrtm(inv_D) #次数行列の逆行列の平方根



#----------------------------シミュレーション---------------------------------

fig = plt.figure()
images = []

for k in range(t):
    y_re=y_diff(k) #実軸
    y_im=-1j*y_diff(k) #虚軸

    image=plt.bar(n_list,y_re,color="blue",width=1.0)
    images.append(image)

ani = animation.ArtistAnimation(fig, images, interval=300)
plt.show()
#ani.save("y_diff(t)_im.gif", writer="imagemagick")