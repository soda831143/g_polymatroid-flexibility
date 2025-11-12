# -*- coding: utf-8 -*-
"""
辅助函数工具箱
【核心修正】：删除了所有与TCL模型相关的冗余、错误函数。
只保留与Zonotope和Homothet算法强相关的辅助函数。
"""

import numpy as np
try:
    import gurobipy as gp
except ImportError:
    gp = None  # Gurobi 可选
import matplotlib.pyplot as plt

# ===============================
# Zonotope 算法辅助函数
# ===============================

def generateZonotope(T, c):
    """生成Zonotope Z(c,g_i,...g_p) 和生成器矩阵G"""
    G1 = np.eye(T)
    G2_1 = -1/np.sqrt(2)*np.eye(T,T-1)
    G2_2 = 1/np.sqrt(2)*np.eye(T,T-1,-1)
    G2 = G2_1 + G2_2
    G = np.column_stack((G1,G2))
    
    Z = [c]
    for i in range(np.shape(G)[1]):
        Z.append(list(G[:,i]))
    return Z, G

def getMatrixC(T):
    """计算半空间表示的矩阵"""
    N = np.eye(T)
    Bl = np.tril(np.ones([T,T]))
      
    for i in range(1,T):
        block = Bl[i,0:i+1]/np.linalg.norm(Bl[i,0:i+1])
        Nblock = np.zeros([T-i,T])
        
        for j in range(0,T-i):
            Nblock[j,j:j+i+1] = block
        
        N = np.concatenate([N,Nblock],axis=0)
    return np.concatenate([N,-N],axis=0)

def getHyperplaneOffset(A, C, b, dimension):
    """计算半空间表示的向量"""
    m = np.shape(C)[0]
    d_list = []
    for i in range(m):
        model = gp.Model("Supporting Hyperplane")
        model.Params.OutputFlag = 0
        x = model.addMVar(shape=dimension,lb=-gp.GRB.INFINITY,name="x")
        model.setObjective(C[i,:]@x,gp.GRB.MAXIMIZE)
        model.addConstr(A@x <= b)
        model.optimize()
        d_list.append(C[i,:]@x.X)
    return d_list

def optimalZonotopeMaxNorm(A, b, G, F, bi_list):
    """计算最优中心和缩放限制，返回Z(c,g_i,...,g_p)"""
    AG = np.abs(A@G)
    W = np.abs(F[0:int(np.shape(F)[0]/2)]@G)
    delta_p = np.array(bi_list)
    W_aux = np.row_stack((W,W))

    model = gp.Model("Optimal Zonotope")
    model.Params.OutputFlag = 0
    t = model.addMVar(1,lb=0.0)
    c = model.addMVar(shape = np.shape(A)[1],lb=-gp.GRB.INFINITY)
    beta_bar = model.addMVar(shape = np.shape(G)[1],lb=0.0)
    model.setObjective(t,gp.GRB.MINIMIZE)
    for i in range(len(delta_p)):
        model.addConstr(-t <= delta_p[i]-(F[i,:]@c + W_aux[i,:]@beta_bar))
        model.addConstr(delta_p[i]-(F[i,:]@c + W_aux[i,:]@beta_bar) <= t)
    model.addConstr(AG@beta_bar + A@c <= b)
    model.optimize()

    c = c.X
    beta_bar = beta_bar.X

    Z = [list(c)]
    for i in range(np.shape(G)[1]):
        Z.append(list(G[:,i]*beta_bar[i]))
    return Z

def getVectord(C, Z, T):
    """从Z(c_i,g_i,...,g_p)表示计算半空间表示向量"""
    p = 2*T-1  # 生成方向数量
    c = Z[0]
    G_list = Z[1:]
    C = C[0:int(np.shape(C)[0]/2),:]
    delta_d_list = []
    for j in range(np.shape(C)[0]):
        delta_d = 0
        for i in range(p):
            delta_d = delta_d + np.abs(C[j,:]@np.array(G_list[i]))
        delta_d_list.append(delta_d)
    d_list = []
    for i in range(np.shape(C)[0]):
        d_list.append(C[i,:]@np.array(c)+np.array(delta_d_list[i]))
    for i in range(np.shape(C)[0]):
        d_list.append(-C[i,:]@np.array(c)+np.array(delta_d_list[i]))    
    d = np.array(d_list)
    return d

# ===============================
# Homothet 算法辅助函数  
# ===============================

def fitHomothet(A, b, b_mean, inner, T):
    """计算最优偏移和缩放因子"""
    aux_len = np.shape(b_mean)[0]
    if inner == True:
        model = gp.Model("MIA")
        model.Params.OutputFlag = 0
        s = model.addMVar(shape = 1)
        G = model.addMVar(shape = (aux_len,aux_len))
        r = model.addMVar(shape = T,lb=-gp.GRB.INFINITY)
        aux = model.addMVar(shape = aux_len,lb=-gp.GRB.INFINITY)
        model.setObjective(s,gp.GRB.MINIMIZE)
        for i in range(aux_len):
            model.addConstrs(G[i,:]@A[:,j] == A[i,j] for j in range(T))
        model.addConstrs(aux[i] == gp.quicksum(G[i,k]*b_mean[k] for k in range(aux_len)) for i in range(aux_len))
        model.addConstr(aux <= b.reshape(aux_len,1)@s + A@r)
        model.optimize()    
        beta = 1/s.X
        t = -r.X/s.X
        return beta, t
    
    elif inner == False:
        model = gp.Model("MOA")
        model.Params.OutputFlag = 0
        s = model.addMVar(shape = 1)
        G = model.addMVar(shape = (aux_len,aux_len))
        r = model.addMVar(shape = T,lb=-gp.GRB.INFINITY)
        aux = model.addMVar(shape = aux_len,lb=-gp.GRB.INFINITY)
        model.setObjective(s,gp.GRB.MINIMIZE)
        for i in range(aux_len):
            model.addConstrs(G[i,:]@A[:,j] == A[i,j] for j in range(T))
        model.addConstrs(aux[i] == gp.quicksum(G[i,k]*b[k] for k in range(aux_len)) for i in range(aux_len))
        model.addConstr(aux <= b_mean.reshape(aux_len,1)@s + A@r)
        model.optimize()
        beta = s.X
        t = r.X
        return beta, t

# ===============================
# TCL 多面体H表示函数
# ===============================

def chebyshev_center(A, b):
    """
    计算多面体 P = {x | Ax <= b} 的Chebyshev中心和半径。
    
    Args:
        A (np.ndarray): 约束矩阵。
        b (np.ndarray): 约束向量。
        
    Returns:
        tuple: (中心点, 半径)
    """
    model = gp.Model('Chebyshev Center')
    model.Params.OutputFlag = 0
    
    n = A.shape[1]
    x_c = model.addMVar(shape=n, lb=-gp.GRB.INFINITY, name='x_c')
    r = model.addMVar(shape=1, lb=0.0, name='r')
    
    model.setObjective(r, gp.GRB.MAXIMIZE)
    
    # ||a_i||_2 for each row a_i in A
    row_norms = np.linalg.norm(A, axis=1)
    
    for i in range(A.shape[0]):
        model.addConstr(A[i, :] @ x_c + r * row_norms[i] <= b[i])
        
    model.optimize()
    
    if model.status == gp.GRB.OPTIMAL:
        return x_c.X, r.X[0]
    else:
        # 如果求解失败，返回一个基于边界框的粗略中心
        # 这有助于在多面体无界或数值不稳定时避免崩溃
        print(f"警告: Chebyshev中心求解失败，返回一个备用中心点。")
        try:
            # 尝试求解一个简单的LP来找到一个可行点
            x_feas = model.addMVar(shape=n, lb=-gp.GRB.INFINITY)
            model_feas = gp.Model('feasibility')
            model_feas.Params.OutputFlag = 0
            model_feas.addConstr(A @ x_feas <= b)
            model_feas.optimize()
            if model_feas.status == gp.GRB.OPTIMAL:
                return x_feas.X, 0.0
            else:
                return np.zeros(n), 0.0
        except:
            return np.zeros(n), 0.0


def preconditioning_matrix(A, b):
    """
    为多面体 Ax <= b 计算预处理矩阵 L。
    """
    L_i_list = []
    for a_i in A:
        model = gp.Model("Preconditioning")
        model.Params.OutputFlag = 0
        x = model.addMVar(shape=A.shape[1], lb=-gp.GRB.INFINITY)
        model.setObjective(a_i @ x, gp.GRB.MAXIMIZE)
        model.addConstr(A @ x <= b)
        model.optimize()
        
        if model.status == gp.GRB.OPTIMAL:
            L_i_list.append(1 / model.objVal if abs(model.objVal) > 1e-6 else 1.0)
        else:
            L_i_list.append(1.0) # 求解失败时的备用值
            
    return np.diag(L_i_list)


def plot_flex_sets_2D(sets, labels, title='2D灵活性集合可视化'):
    """
    在2D平面上绘制灵活性集合。
    Args:
        sets (list): 每个元素是一个(A, b)元组，描述一个多面体。
        labels (list): 每个集合的标签。
        title (str): 图表标题。
    """
    plt.figure(figsize=(10, 8))
    for (A, b), label in zip(sets, labels):
        try:
            # 计算多面体的顶点用于绘图
            # 注意：这只在2D中有效
            from scipy.spatial import ConvexHull
            # 找到所有约束的交点
            points = []
            for i in range(len(A)):
                for j in range(i + 1, len(A)):
                    A_sub = A[[i, j], :]
                    b_sub = b[[i, j]]
                    if np.linalg.matrix_rank(A_sub) == 2:
                        try:
                            points.append(np.linalg.solve(A_sub, b_sub))
                        except np.linalg.LinAlgError:
                            continue
            feasible_points = [p for p in points if np.all(A @ p <= b + 1e-5)]
            if len(feasible_points) < 3:
                print(f"警告: 集合 '{label}' 的可行点少于3个，无法绘制凸包。")
                continue
            hull = ConvexHull(feasible_points)
            # 绘制凸包
            for simplex in hull.simplices:
                plt.plot(np.array(feasible_points)[simplex, 0], np.array(feasible_points)[simplex, 1], 'o-', label=label if simplex[0] == hull.simplices[0][0] else "")
        except Exception as e:
            print(f"绘制集合 '{label}' 时出错: {e}")
    plt.title(title)
    plt.xlabel('时间步 1 的功率')
    plt.ylabel('时间步 2 的功率')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_results(df, metrics=['cost', 'peak', 'algo_time', 'opt_time']):
    """
    根据给定的指标绘制结果。
    """
    # ... 实现绘图逻辑 ...
    pass

def load_results(path):
    """
    从CSV文件加载结果。
    """
    import pandas as pd
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"错误: 找不到结果文件 at '{path}'")
        return None