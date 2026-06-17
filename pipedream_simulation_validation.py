#This code is used to compare model output between Pipedream and Epanet 2025.03.03

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
 

def run_pipedream_simulation(inp, t_run=None, dt=None, banded=False, num_iter=40, use_tank_init_cond=None):
    #Input path
    inp_file=inp  
    
    #Parameter for how many hours to run the model
    wn = wntr.network.WaterNetworkModel(inp)
    node_names = wn.node_name_list   
    link_names = wn.link_name_list
    
    wntr_time=[]
    
    results_dict={}
    
    #Run model in wntr
    # Are these lines necessary?
    wn.options.time.hydraulic_timestep=dt
    wn.options.time.report_timestep=dt    
    wn.options.time.duration=t_run*dt
    
    sim = wntr.sim.EpanetSimulator(wn)
    
    t1=time.time()
    results = sim.run_sim()
    t2=time.time()
    
    wntr_time.append(t2-t1)
    
    #Does not need to be in the loop since the model formulation does not vary by timestep
    superjunctions, superlinks, orifices, pumps, prvs, H_bc, Q_in, pats, mult_df, tank_min, tank_max, tank_dict, time_controls_compiled, events_controls_pairs = pdu.wntr_2_pd(wn, t_run, dt)
    
    t3 = time.time()
    #print('model formulation time: ', t3-t2)
    mult_df['-'] = 0.
    #mult_df.index = mult_df.index.tolist()
    multipliers = mult_df
    
    #%% Run Model- Baseline
    
    # Specify number of internal links in each superlink and timestep size in seconds
    internal_links = 1
    num_tanks=wn.num_tanks
    num_valves=1
    u_o = np.ones(num_tanks+num_valves)
    #u_p = np.ones(wn.num_pumps)
    u_p = np.array([1., 1., 0., 0., 0., 0., 1., 1., 0., 1., 0.])
    u_prv = np.ones(wn.num_valves)
    u_l = np.ones(wn.num_pipes)
    #superlinks.loc[superlinks['name'] == 'v2', 'g1'] = 1e-6
    
    model = SuperLink(superlinks, superjunctions,  
                      internal_links=internal_links, orifices=orifices, pumps=pumps, prvs = prvs, auto_permute=banded)
    
    is_tank = model.superjunctions['tank'].values
    is_tank_fake_node = model.superjunctions['tank_fake_node'].values
    tank_min = model.superjunctions['tank_min'].values
    tank_max = model.superjunctions['tank_max'].values 
    
    #Initiate model
    # Spin up model at small timestep to 'settle' initial states
    
    t3=time.time()
    Q_in_t = -(model.superjunctions['demand_pattern'].map(multipliers.loc[0]).fillna(0.) * model.superjunctions['base_demand']).values
    model.spinup(n_steps=10, dt=60, Q_in=Q_in_t, H_bc=H_bc, u_l=u_l, u_o=u_o, u_p=u_p, u_prv=u_prv, banded=banded)
    t3_5=time.time()
    #print('model spinup time: ', t3_5-t3)
    
    H = []
    Q = []
    Q_o = []
    Q_pump = []
    Q_prv = []
    Q_in_all = []
    t=[]
    alpha_prv = []
    beta_prv = []
    chi_prv = []
        
    #Run model for 24 hours
    # While time is less than 24 hours
    while model.t < (t_run * 3600):    
        
        hour=model.t/3600
        #print('hour ', hour)
        j=int(np.floor(model.t//wn.options.time.pattern_timestep))
#        if 'Net1' in inp:
#            j = int(np.floor(model.t/3600)) //2
        Q_in_t = -(model.superjunctions['demand_pattern'].map(multipliers.loc[j]).fillna(0.) * model.superjunctions['base_demand']).values
        print(j, model.t)
        Q_in_all.append(Q_in_t)
        H_bc_t = H_bc
        
        # Set tank initial conditions
        if use_tank_init_cond:
            if model.t == 0:
                H_bc_t[is_tank] = model.superjunctions['h_0'].values[is_tank] + model.superjunctions['z_inv'].values[is_tank]
                model.bc[is_tank] = True
                H_bc_t_0=H_bc_t
            else:
                model.bc[is_tank] = False
        
        # Tanks
        #u_o[:num_tanks] = ((model.H_j[is_tank] > tank_min[is_tank]) & (model.H_j[is_tank] < tank_max[is_tank])).astype(np.float64)
        u_o[:num_tanks] = ((model.H_j[is_tank_fake_node] > tank_min[is_tank_fake_node]+0.0) & (model.H_j[is_tank_fake_node] < tank_max[is_tank_fake_node]-0.0)).astype(np.float64) #0.1 buffer before
    
                
        #print(H_bc_t, model.bc)
        # Check control rule status
        # open link --> 1, close link --> 0
                
        # Event based controls -- only assuming pump - tank control rules for here. Modify for NWC
        # this is also for > upper limit
        
        # for key in events_controls_pairs.keys():
        #     node = events_controls_pairs[key]['Node']
        #     link = events_controls_pairs[key]['Link']
        #     node_id = list(model.superjunctions.loc[model.superjunctions['name']==node,'id'])[0]
        #     if link in wn.pump_name_list:
        #         pump_id = model.pumps.loc[model.pumps['name']==link,'id'].values
                
        #     if model.H_j[node_id] > events_controls_pairs[key]['Upper lim']:
        #         u_p[pump_id] = events_controls_pairs[key]['Upper lim stat']
        #     if model.H_j[node_id] < events_controls_pairs[key]['Lower lim']:
        #         u_p[pump_id] = events_controls_pairs[key]['Lower lim stat']
        #%% manually set the control rules
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
        if model.H_j[T1_id] > 71.5 + 4.5 - 0.0:
            u_p[PU2_id] = 0
            
        # Tank 2
        if model.H_j[T2_id] < 65 + 0.5 + 0.0:
            u_o[7] = 1
        if model.H_j[T2_id] > 65 + 5.5 - 0.0:
            u_o[7] = 0
        
        # Tank 3
        if model.H_j[T3_id] < 112.9 + 3 + 0.0:
            u_p[PU4_id] = 1
        if model.H_j[T3_id] > 112.9 + 5.3 - 0.2:
            u_p[PU4_id] = 0
       
        if model.H_j[T3_id] < 112.9 + 1:
            u_p[PU5_id] = 1
        if model.H_j[T3_id] > 112.9 + 3.5:
            u_p[PU5_id] = 0
        
        # Tank 4
        if model.H_j[T4_id] < 132.5 + 2 - 2: # adjust
            u_p[PU6_id] = 1
        if model.H_j[T4_id] > 132.5 + 3.5 + 2:
            u_p[PU6_id] = 0
            
        if model.H_j[T4_id] < 132.5 + 3 + 0.0: # adjust
            u_p[PU7_id] = 1
        if model.H_j[T4_id] > 132.5 + 4.5 - 0.25: # adjust
            u_p[PU7_id] = 0
            
        # Tank 5
        if model.H_j[T5_id] < 105.8 + 1.5:
            u_p[PU8_id] = 1
        if model.H_j[T5_id] > 105.8 + 3.5 + 0.35:
            u_p[PU8_id] = 0
            
        # Tank 7
        if model.H_j[T7_id] < 102 + 2.5:
            u_p[PU10_id] = 1
        if model.H_j[T7_id] > 102 + 4.8 - 0.5: # adjust
            u_p[PU10_id] = 0
            
        if model.H_j[T7_id] < 102 + 1:
            u_p[PU11_id] = 1
        if model.H_j[T7_id] > 102 + 3:
            u_p[PU11_id] = 0
        #%%
        #print(j, 'u_o', u_o)
        #Run model
        model.step(dt=dt, H_bc = H_bc_t, Q_in=Q_in_t, u_l=u_l, u_o=u_o, u_p=u_p, u_prv=u_prv,
                    banded=banded, num_iter=num_iter, head_tol=0.0001) # initial conditions
        #%%
        #Extract results at each timestep
        H.append(model.H_j.copy())
        Q.append(model.Q_ik.copy())
        Q_o.append(model.Q_o.copy())
        Q_pump.append(model.Q_p.copy())
        Q_prv.append(model.Q_prv.copy())
        #dem_source.append(Q_in_t[26].copy())
        t.append(model.t)
        
        #print('valve flow', model.Q_o)
        alpha_prv.append(model._alpha_prv.copy())
        beta_prv.append(model._beta_prv.copy())
        chi_prv.append(model._chi_prv.copy())
    
    t4=time.time()
    #print('simulation run time: ', t4-t3_5)
    
    pd_time_spin= t3_5-t3
    pd_time_rest = t4-t3_5
    pd_time_tot = t4-t3
    
    
    H = np.vstack(H)
    Q = np.vstack(Q)
    Q_o = np.vstack(Q_o)
    Q_pump = np.vstack(Q_pump)
    Q_prv = np.vstack(Q_prv)
    #dem_source=np.vstack(dem_source)
    t=np.vstack(t)
    
    alpha_prv = np.vstack(alpha_prv)
    beta_prv = np.vstack(beta_prv)
    chi_prv = np.vstack(chi_prv)
        
    
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
    t5 = time.time()
    #print('finishing up time: ', t5-t4)
    
    return H_df, Q_df, Q_o, Q_pump, Q_prv, model, Q_in_all_df , pumps, superjunctions, orifices, superlinks, prvs, flip_mult
    
