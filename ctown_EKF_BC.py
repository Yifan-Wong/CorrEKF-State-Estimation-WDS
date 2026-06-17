#This code is used to compare model output between Pipedream and Epanet

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wntr
import scipy as sc
import networkx as nx
import networkx.drawing.nx_pylab as nxp
from pipedream_solver.hydraulics import SuperLink
from pipedream_solver.simulation import Simulation
from pipedream_solver.nutils import interpolate_sample
import random
import time
# import pipedream_utility_v3 as pdu
import pipedream_utility as pdu
from matplotlib.ticker import FormatStrFormatter


#Don't show future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
 
def create_full_matrix(model, dt, Q_in):  # yifan delete Q_Ik
    g = 9.81
    ndim = model.M + (model.NK * 5) + model.n_o + model.n_p + model.n_prv
    A_1 = np.zeros((ndim, ndim))
    A_2 = np.zeros((ndim, ndim))
    B = np.eye(ndim)
    D = np.zeros(ndim)
    u = np.zeros(ndim)
    
    uk_offset = model.M
    dk_offset = uk_offset + model.NK
    o_offset = dk_offset + model.NK
    p_offset = o_offset + model.n_o
    prv_offset = p_offset + model.n_p
    Ik_offset = prv_offset + model.n_prv
    ik_offset = Ik_offset + model._I.size
    
    j = np.arange(model.M, dtype=np.int64)
    uk = np.arange(uk_offset, uk_offset + model.NK)
    dk = np.arange(dk_offset, dk_offset + model.NK)
    o = np.arange(o_offset, o_offset + model.n_o)
    p = np.arange(p_offset, p_offset + model.n_p)
    prv = np.arange(prv_offset, prv_offset + model.n_prv)
    Ik = Ik_offset + model._I
    ik = ik_offset + model._i
    
    cont_j_j = model.A_sj / dt
    cont_j_uk = np.zeros(model.M)
    cont_j_dk = np.zeros(model.M)
    np.add.at(cont_j_uk, model._J_uk, model._B_uk * model._dx_uk * model._theta_uk / 2 / dt)
    np.add.at(cont_j_dk, model._J_dk, model._B_dk * model._dx_dk * model._theta_dk / 2 / dt)
    A_1[j, j] = cont_j_j + cont_j_uk + cont_j_dk
    A_1[model._J_uk, uk] = 1.
    A_1[model._J_dk, dk] = -1.
    A_1[model._J_uo, o] = 1.
    A_1[model._J_do, o] = -1.
    # yifan add mass balance for Pump
    A_1[model._J_up, p] = 1.
    A_1[model._J_dp, p] = -1.
    # yifan add mass balance for PRV
    A_1[model._J_uprv, prv] = 1.
    A_1[model._J_dprv, prv] = -1.
    
    A_2[j, j] = cont_j_j + cont_j_uk + cont_j_dk
    u[j] = Q_in
    
    A_1[uk, uk] = -model._kappa_uk
    A_1[uk, model._J_uk] = -model._lambda_uk
    A_1[uk, Ik_offset + model._I_1k] = 1.
    A_2[uk, uk] = model._dx_uk / g / model._A_uk / dt
    D[uk] = model._dx_uk * model._S_o_uk - model._theta_uk * model._z_inv_uk
    
    A_1[dk, dk] = -model._kappa_dk
    A_1[dk, model._J_dk] = -model._lambda_dk
    A_1[dk, Ik_offset + model._I_Np1k] = 1
    A_2[dk, dk] = -model._dx_dk / g / model._A_dk / dt
    D[dk] = - model._dx_dk * model._S_o_dk - model._theta_dk * model._z_inv_dk
    
    A_1[o, o] = 1.
    A_1[o, model._J_uo] = -model._alpha_o
    A_1[o, model._J_do] = -model._beta_o
    D[o] = model._chi_o
    # yifan add pump and prv
    A_1[p, p] = 1.
    A_1[p, model._J_up] = -model._alpha_p
    A_1[p, model._J_dp] = -model._beta_p
    D[p] = model._chi_p
    
    A_1[prv, prv] = 1.
    A_1[prv, model._J_uprv] = -model._alpha_prv
    A_1[prv, model._J_dprv] = -model._beta_prv
    D[prv] = model._chi_prv
    
    A_1[Ik, Ik] = model._E_Ik
    # Check this
    A_1[Ik_offset + model._I_1k, uk] = -1.
    A_1[Ik_offset + model._I_Np1k, dk] = 1.
    A_1[Ik_offset + model._Ik, ik] = 1.
    A_1[Ik_offset + model._Ip1k, ik] = -1.
    # Fix this
    A_2[Ik, Ik] = model._D_Ik / model.states['h_Ik']   # yifan delete Q_Ik  (model._D_Ik - Q_Ik) / model.states['h_Ik']
    # u[Ik] = Q_Ik  # yifan delete Q_Ik
    
    A_1[ik, ik] = model._b_ik
    A_1[ik_offset + model._i_1k, uk] = model._a_ik
    A_1[ik_offset + model._i_nk, dk] = model._c_ik
    # Need to extend this
    A_1[ik, Ik_offset + model._Ip1k] = g * model._A_ik
    A_1[ik, Ik_offset + model._Ik] = -g * model._A_ik
    A_2[ik, ik] = model._dx_ik / dt
    D[ik] = g * model._A_ik * model._S_o_ik * model._dx_ik

    # yifan add
    variables_to_concatenate = [model.states[var] for var in ['H_j', 'Q_uk', 'Q_dk', 'Q_o', 'Q_p', 'Q_prv', 'h_Ik', 'Q_ik'] if var in model.states]
    x_t = np.concatenate(variables_to_concatenate)

    x_tp1 = np.concatenate([model.H_j, model.Q_uk, model.Q_dk, model.Q_o, model.Q_p, model.Q_prv,
                            model.h_Ik, model.Q_ik])
    
    return A_1, A_2, B, D, x_t, x_tp1, u, j, uk, dk, o, p, prv, Ik, ik, ik_offset

def apply_EKF_BC(inp, imp_list, msmts, flow_list, bc_measurements, measurements, Qcov, Rcov, t_run=None, dt=None, banded=False, num_iter=40, use_tank_init_cond=None):
    
    wn = wntr.network.WaterNetworkModel(inp)
    
    superjunctions, superlinks, orifices, pumps, prvs, H_bc, Q_in, pats, mult_df, tank_min, tank_max, tank_dict, time_controls_compiled, events_controls_pairs = pdu.wntr_2_pd(wn, t_run, dt)
    
    mult_df['-'] = 0.
    multipliers = mult_df
    
    internal_links = 1
    num_tanks = wn.num_tanks
    num_valves = 1
    u_o = np.ones(num_tanks+num_valves)
    u_p = np.array([1., 1., 0., 1., 0., 0., 1., 1., 0., 1., 0.])
    u_prv = np.ones(wn.num_valves)
    u_l = np.ones(wn.num_pipes)
    
    model = SuperLink(superlinks, superjunctions,  
                      internal_links=internal_links, orifices=orifices, pumps=pumps, prvs = prvs, auto_permute=banded)
    
    is_tank = model.superjunctions['tank'].values
    is_tank_fake_node = model.superjunctions['tank_fake_node'].values
    tank_min = model.superjunctions['tank_min'].values
    tank_max = model.superjunctions['tank_max'].values 
    #%%
    m = len(msmts) + len(flow_list)
    ndim = model.M + (model.NK * 5) + model.n_o + model.n_p + model.n_prv
    ik_offset =  model.M + (model.NK * 4) + model.n_o + model.n_p + model.n_prv
    
    head_sensor_matrix_location = np.array(msmts)
    flow_sensor_matrix_location = ik_offset + np.array(flow_list)
    
    P_x_k_k = Qcov.copy()
    C_k = np.zeros((m, ndim))
    #Enter the sensor locations
    for i, col in enumerate(head_sensor_matrix_location):
        C_k[i, col] = 1
    for i, col in enumerate(flow_sensor_matrix_location):
        C_k[len(msmts) + i, col] = 1
    #%%
    Q_in_t = -(model.superjunctions['demand_pattern'].map(multipliers.loc[0]).fillna(0.) * model.superjunctions['base_demand']).values
    model.spinup(n_steps=10, dt=60, Q_in=Q_in_t, H_bc=H_bc, u_l=u_l, u_o=u_o, u_p=u_p, u_prv=u_prv, banded=banded)
    
    H = []
    Q = []
    Q_o = []
    Q_pump = []
    Q_prv = []
    Q_in_all = []
    t=[]
    P_xk1k = []
    P_xk1k1 = []
    #P_xkk.append(P_x_k_k.copy())
    
    #Run model for 24 hours
    # While time is less than 24 hours
    while model.t < (t_run * 3600):    
        
        hour=model.t/3600
        j=int(np.floor(model.t//wn.options.time.pattern_timestep))
        Q_in_t = -(model.superjunctions['demand_pattern'].map(multipliers.loc[j]).fillna(0.) * model.superjunctions['base_demand']).values
        print(j, model.t)
        Q_in_all.append(Q_in_t)
        H_bc_t = H_bc
        
        for name in imp_list:
            H_bc_t[int(list(superjunctions['name']).index(name))] = bc_measurements[name][model.t]
        
        # Set tank initial conditions
        # if use_tank_init_cond:
        #     if model.t == 0:
        #         H_bc_t[is_tank] = model.superjunctions['h_0'].values[is_tank] + model.superjunctions['z_inv'].values[is_tank]
        #         model.bc[is_tank] = True
        #         H_bc_t_0=H_bc_t
        #     else:
        #         model.bc[is_tank] = False
        
        # Tanks
        u_o[:num_tanks] = ((model.H_j[is_tank_fake_node] > tank_min[is_tank_fake_node]+0.0) & (model.H_j[is_tank_fake_node] < tank_max[is_tank_fake_node]-0.0)).astype(np.float64) # 
        
        # manually set the control rules
        # open link --> 1, close link --> 0
        T1_id = list(model.superjunctions.loc[model.superjunctions['name']=='T1','id'])[0]
        T2_id = list(model.superjunctions.loc[model.superjunctions['name']=='T2','id'])[0]
        T3_id = list(model.superjunctions.loc[model.superjunctions['name']=='T3','id'])[0]
        T4_id = list(model.superjunctions.loc[model.superjunctions['name']=='T4','id'])[0]
        T5_id = list(model.superjunctions.loc[model.superjunctions['name']=='T5','id'])[0]
        T7_id = list(model.superjunctions.loc[model.superjunctions['name']=='T7','id'])[0]
        #v2_id = list(model.superlinks.loc[model.superlinks['name']=='v2','id'])[0]
        PU1_id = model.pumps.loc[model.pumps['name']=='PU1','id'].values
        PU2_id = model.pumps.loc[model.pumps['name']=='PU2','id'].values
        PU4_id = model.pumps.loc[model.pumps['name']=='PU4','id'].values
        PU5_id = model.pumps.loc[model.pumps['name']=='PU5','id'].values
        PU6_id = model.pumps.loc[model.pumps['name']=='PU6','id'].values
        PU7_id = model.pumps.loc[model.pumps['name']=='PU7','id'].values
        PU8_id = model.pumps.loc[model.pumps['name']=='PU8','id'].values
        PU10_id = model.pumps.loc[model.pumps['name']=='PU10','id'].values
        PU11_id = model.pumps.loc[model.pumps['name']=='PU11','id'].values
        
        # Tank 1
        if model.H_j[T1_id] < 71.5 + 4:
            u_p[PU1_id] = 1
        if model.H_j[T1_id] > 71.5 + 6.3:
            u_p[PU1_id] = 0
        
        if model.H_j[T1_id] < 71.5 + 1:
            u_p[PU2_id] = 1
        if model.H_j[T1_id] > 71.5 + 4.5 - 0.2:
            u_p[PU2_id] = 0
            
        # Tank 2
        if model.H_j[T2_id] < 65 + 0.5 + 0.4:
            u_o[7] = 1
        if model.H_j[T2_id] > 65 + 5.5 + 0.1:
            u_o[7] = 0
        
        # Tank 3
        if model.H_j[T3_id] < 112.9 + 3:
            u_p[PU4_id] = 1
        if model.H_j[T3_id] > 112.9 + 5.3:
            u_p[PU4_id] = 0
       
        if model.H_j[T3_id] < 112.9 + 1:
            u_p[PU5_id] = 1
        if model.H_j[T3_id] > 112.9 + 3.5:
            u_p[PU5_id] = 0
        
        # Tank 4
        if model.H_j[T4_id] < 132.5 + 2 - 2: # yifan adjust
            u_p[PU6_id] = 1
        if model.H_j[T4_id] > 132.5 + 3.5 + 2:
            u_p[PU6_id] = 0
            
        if model.H_j[T4_id] < 132.5 + 3: # yifan adjust
            u_p[PU7_id] = 1
        if model.H_j[T4_id] > 132.5 + 4.5 - 0.3: # yifan adjust
            u_p[PU7_id] = 0
            
        # Tank 5
        if model.H_j[T5_id] < 105.8 + 1.5:
            u_p[PU8_id] = 1
        if model.H_j[T5_id] > 105.8 + 3.5 + 0.15:
            u_p[PU8_id] = 0
            
        # Tank 7
        if model.H_j[T7_id] < 102 + 2.5:
            u_p[PU10_id] = 1
        if model.H_j[T7_id] > 102 + 4.8 - 0.5: # yifan adjust
            u_p[PU10_id] = 0
            
        if model.H_j[T7_id] < 102 + 1:
            u_p[PU11_id] = 1
        if model.H_j[T7_id] > 102 + 3:
            u_p[PU11_id] = 0
        #%
        #print(j, 'u_o', u_o)
        #Run model
        model.step(dt=dt, H_bc = H_bc_t, Q_in=Q_in_t, u_l=u_l, u_o=u_o, u_p=u_p, u_prv=u_prv,
                    banded=banded, num_iter=num_iter, head_tol=0.0001) # initial conditions
        
        Z = measurements.loc[model.t].values
        
        A_1, A_2, B, D, x_t, x_tp1, u, j, uk, dk, o, p, prv, Ik, ik, ik_offset = create_full_matrix(model, dt, Q_in_t)
        
        for name in imp_list:
            bc_location = int(list(superjunctions['name']).index(name))
            A_1[bc_location, :] = 0
            A_1[bc_location, bc_location] = 1
            A_2[bc_location, :] = 0
            D[bc_location] = H_bc_t[bc_location]
            u[bc_location] = 0
        
        b = A_2 @ x_t + D + u
        I = np.eye(A_1.shape[0])
        A_1_inv = np.linalg.inv(A_1)
        x_k1_k = A_1_inv @ b
        P_x_k1_k = A_1_inv @ A_2 @ P_x_k_k @ A_2.T @ A_1_inv.T + B @ Qcov @ B.T
        L_x_k1 = P_x_k1_k @ C_k.T @ np.linalg.inv((C_k @ P_x_k1_k @ C_k.T) + Rcov)
        P_x_k1_k1 = (I - L_x_k1 @ C_k) @ P_x_k1_k # might cause negative error cov
        gain = L_x_k1 @ (Z - C_k @ x_k1_k)
        x_hat = x_k1_k + gain
        P_x_k_k = P_x_k1_k1
        
        model.H_j[:] = x_hat[j]
        model.Q_uk[:] = x_hat[uk]
        model.Q_dk[:] = x_hat[dk]
        model.Q_o[:] = x_hat[o]
        model.Q_p[:] = x_hat[p]
        model.Q_prv[:] = x_hat[prv]
        model.h_Ik[:] = x_hat[Ik]
        model.Q_ik[:] = x_hat[ik]
        #%
        #Extract results at each timestep
        H.append(model.H_j.copy())
        Q.append(model.Q_ik.copy())
        Q_o.append(model.Q_o.copy())
        Q_pump.append(model.Q_p.copy())
        Q_prv.append(model.Q_prv.copy())
        t.append(model.t)
        P_xk1k.append(P_x_k1_k.copy())
        P_xk1k1.append(P_x_k1_k1.copy())
    
    H = np.vstack(H)
    Q = np.vstack(Q)
    Q_o = np.vstack(Q_o)
    Q_pump = np.vstack(Q_pump)
    Q_prv = np.vstack(Q_prv)
    #dem_source=np.vstack(dem_source)
    t=np.vstack(t)
    P_xk1k = np.stack(P_xk1k, axis=0)
    P_xk1k1 = np.stack(P_xk1k1, axis=0)
    
    #Sample down the Q matrix to only every column from a new link, not each sub-link
    #i.e. if there are 12 internal links in each superlink, then each of the first 
    #12 columns in q will basically be the same
    
    n_superlinks,x=superlinks.shape
    Q_superlinks=Q[:,0:n_superlinks*internal_links:internal_links]
    
    #put H and Q into a dataframe
    #Unscramble the head matrix
    perm_inv = np.argsort(model.permutations)
    H = H[:, perm_inv]
    
    #Do not use model.superjunctions because I want to use the head matrix in the same order
    #as the original (unpermuted) superjunctions DF because then the columns correspond
    #to the wntr results
    H_df=pd.DataFrame(columns=superjunctions['name'],index=np.arange(0,t_run*3600,dt),data=H)
    Q_df=pd.DataFrame(columns=model.superlinks['name'],index=np.arange(0,t_run*3600,dt),data=Q_superlinks)
    Q_o=pd.DataFrame(columns=model.orifices['name'],index=np.arange(0,t_run*3600,dt),data=Q_o)
    Q_pump=pd.DataFrame(columns=model.pumps['name'],index=np.arange(0,t_run*3600,dt),data=Q_pump)
    Q_prv=pd.DataFrame(columns=model.prvs['name'],index=np.arange(0,t_run*3600,dt),data=Q_prv)
    
    #Figure out which, if any, of the in/out nodes of the links got reversed
    #due to changes in elevation
    
    from_node=model.superlinks['sj_0'].to_numpy()
    
    #Convert back to the nodes of the un-permuted superjunction indicies 
    from_node_orig=[]
    for i in range(len(from_node)):
        #find the index in perm_inv where it equals the value of from_node
        from_node_orig.append(np.where(perm_inv==from_node[i])[0][0])
    from_node_orig=np.array(from_node_orig)
    
    #True means the nodes got flipped
    flipped=superlinks['sj_0'].to_numpy()!=from_node_orig
    
    flip_mult=np.ones(len(flipped))
    flip_mult[flipped]=-1
    
    Q_df=Q_df*flip_mult
    
    Q_in_all=np.vstack(Q_in_all)
    Q_in_all=Q_in_all[:, perm_inv]
            
    Q_in_all_df=pd.DataFrame(Q_in_all,index=t.flatten())
     
    return H_df, Q_df, Q_pump, Q_prv, model, Q_in_all_df, pumps, superjunctions, orifices, superlinks, prvs, flip_mult, P_xk1k, P_xk1k1     
