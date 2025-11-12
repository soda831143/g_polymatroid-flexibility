#!/usr/bin/env python
# coding: utf-8

# # Packages and Functions

# In[1]:


import os
import socket
import importlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import json
import time
# from tqdm.notebook import tqdm
import multiprocessing

import gurobipy as gp

import lib
# lib.algo_template.algo(data)
# getattr(getattr(lib, "algo_template"), "algo")(data)


# # Configuration

# In[7]:


cfg = {}

# relative (to this file) data path:
cfg['data path'] = "../data/"

# time periods: sampling time = 1/4 h
cfg['periods list'] = [4, 8, 12, 16, 20, 24]  # list of EVEN integer numbers of time periods
# cfg['fade out'] = 0.5  # additional end-portion of data: 0.5 = 50 % more data for optimization at data end
cfg['fade out'] = 0.0  # no fade out

# households:
cfg['households list'] = [2, 6, 10, 20, 30, 40, 50]   # list of numbers of households to be aggregated, <= 200

# batteries:
cfg['S_end_min'] = 0.5  # *S_0
# cfg['S_end_min'] = 0.0  # *S_0

# algorithms: exact Minkowski sum, approximations, benchmark
cfg['algos'] = {
    # --- for COMPARISON and QUALITY CRITERIA:
    'no flexibility':"algo_no_flex",            # x = 0 for benchmarking (KR)
    'exact'         :"algo_exact",              # Minkowski sum (KR)
    # --------------------------------------------------------------
    
    # --- FAST:
    'OA by RHS Summation'             :"algo_Barot_wo_pc",                   # outer (KR)
    'OA by RHS Summation with PC'     :"algo_Barot_w_pc",                    # outer (OZEM)
    'IA with Cuboid Homothets Stage 0':"algo_Union_Homothets_Stage_0",       # inner (OZEM)
    'OA with Battery Homothets'       :"algo_Outer_Homothets",               # outer (OZEM)
    
    'IA with Battery Homothets'                 :"algo_Inner_Homothets",     # inner (OZEM)
    'IA by Battery Homothet Projection with LDR':"algo_Homothet_Projection", # inner (OZEM)
    
    # --- MEDIUM-FAST:
    'IA with Cuboid Homothets Stage 1'   :"algo_Union_Homothets_Stage_1",    # inner (OZEM)
    'IA with Zonotopes $l_\infty$'       :"algo_Zonotope",                   # inner (OZEM)
    'IA with Zonotopes $l_1$'            :"algo_Zonotope_l1",                # inner (OZEM)
    'IA with Zonotopes $l_2$'            :"algo_Zonotope_l2",                # inner (OZEM)
    'IA with Zonotopes weighted'         :"algo_Zonotope_Rel",               # inner (OZEM)
    'IA by Ellipsoid Projection with LDR':"algo_Zhen_Inner",                 # inner (OZEM)
    
    # --- SLOW:
    'IA by Ellipsoid Projection':"algo_Barot_Inner",                         # inner (OZEM)
    
    # ---------------------------------------------------------------
    # --- KR:
    # 'Simple Inner'              :"algo_Simple_Inner",    # inner (KR)
    # 'Klaus inner span'          :"algo_Klaus_Cost",      # inner (OEZM)
    # 'Barot w. pc. KR'           :"algo_Barot_w_pc_KR",   # Barot with preconditioning (KR)    
    #  '...':...   # ...
}

# objectives:
# cfg['objectives'] = ['cost']
# cfg['objectives'] = ['peak'] # peak power
cfg['objectives'] = ['cost', 'peak']

# sampling:
cfg['days'] = [day.strftime(format='%Y-%m-%d')
               for day in pd.date_range(start="2016-01-15", end="2016-12-15", periods=12)]
cfg['samples'] = 10  # number of random villages
# cfg['rng seed'] = None
cfg['rng seed'] = 0

# multiprocessing processes (cores): 1 means no multiprocessing
proc_max = multiprocessing.cpu_count()
cfg['multiprocessing processes'] = min(10, proc_max - 1)

# breakthrough time: in seconds
# cfg['breakthrough time'] =   10.0
cfg['breakthrough time'] = 1*60.0

# verbosity:
# cfg['verbose'] = True
cfg['verbose'] = False

# relative results path:
hostname = socket.gethostname()
cfg['results path trunk 1'] = "../results/processed_data/"

# cfg['results path trunk 2'] = f"benchmark_host/{hostname}/"
# cfg['results path trunk 2'] = f"testing/{hostname}/"
# cfg['results path trunk 2'] = f"no_tail_zero_end/{hostname}/"
# cfg['results path trunk 2'] = f"with_tail_zero_end/{hostname}/"
cfg['results path trunk 2'] = f"no_tail_half_end/{hostname}/"

# cfg['results dir'] = f"{list(cfg['algos'].values())[0]}/"
# cfg['results dir'] = "various_algos_part_4/"
cfg['results dir'] = "various_algos/"

# ----------------------------------
if 1: # ----- ONLY FOR TESTING -----
    print("===!!!=== TEST SETTING ===!!!===")
    cfg['periods list']    = [2, 5, 10]       
    cfg['households list'] = [5, 10, 15]
    cfg['days'] = cfg['days'][5:7]
    cfg['samples']         = 5
    cfg['multiprocessing processes'] = min(10, proc_max - 1) # 3
    cfg['breakthrough time'] = 10 # 1*60
    cfg['results path trunk 2'] = f"testing/{hostname}/"
    cfg['results dir'] = "various_algos/"
# ----------------------------------

cfg['results path'] = cfg['results path trunk 1'] + cfg['results path trunk 2'] + \
                      cfg['results dir']
print(f"results dir      : {cfg['results path trunk 2'] + cfg['results dir']}")
print(f"CPUs             : {cfg['multiprocessing processes']}")
print(f"breakthrough time: {cfg['breakthrough time']/60.0:.2f} minutes for one day calculations")


# In[8]:


if 0:  # print info of days 
    day_name = ['Monday', 'Tuesday', 'Wednesday','Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in cfg['days']:
        print(f"{day} is a {day_name[pd.Timestamp(day).day_of_week]}.") # Monday=0 -> Sunday=6
if 0:  # rng testing
    rng = np.random.default_rng(seed=0)
    rng.uniform()


# # Data
# 
# - **units:**
#     - data: h, kW, kWh, EUR
#      - algo times: seconds
# - **sampling time** = 1/4 h

# ## Prices (EUR/kWh)

# In[9]:


df_prices = pd.read_pickle(cfg["data path"] + 'da_df.pickle')
df_prices = df_prices/1000
df_prices.columns = ["DA_EUR/kWh"]

df_prices.head(3)


# ## Demand (kW)

# In[10]:


household_IDs = pd.read_pickle(cfg["data path"] + "hh_residential_IDs_file1.pickle")
df_hh_list = []
for ID in household_IDs:
    df_hh_list.append(pd.read_pickle(cfg["data path"] + f"hh_df_{ID}.pickle"))
df_demands = pd.concat(df_hh_list, axis=1)
df_demands.columns = [f"HH_{h}_kW" for h in range(len(household_IDs))]

df_demands.iloc[:3,:5]


# ## Batteries

# We used uniformly distributed random variables for $S_{\text{max}} \in [10.5,13.5]$ (kWh), $S_{0} \in [0,10.5]$ (kWh), $x_{\text{max}} \in [4,6]$ (kW), and $x_{\text{min}} \in [-6,-4]$ (kW) following [Tesla PowerWall](https://www.tesla.com/de_at/powerwall).

# In[11]:


bat = {}
bat['S_max_range'] = [10.5, 13.5]  # kWh
bat['S_0_range'  ] = [ 0.0, 10.5]  # kWh
bat['x_max_range'] = [ 4.0,  6.0]  # kW
bat['x_min_range'] = [-6.0, -4.0]  # kW


# # Computations
# 
# algo_time: in seconds

# In[12]:


# stop overall computation time:
t0 = time.time()

# reload local functions in lib directory:
importlib.reload(lib)
importlib.reload(lib.tools)
for algo_module in cfg['algos'].values():
    importlib.reload(getattr(lib, algo_module))

# set RNG seed:
rng = np.random.default_rng(seed=cfg['rng seed'])

# create results dataframe:
n = len(cfg['periods list'])*len(cfg['households list'])*len(cfg['days'])*cfg['samples']*len(cfg['algos'])
columns_1 = ['periods', 'households', 'day', 'sample', 'algo', 'algo_time']
columns_2 = []
for obj in cfg['objectives']:
    columns_2.append(f"{obj}_value")
    columns_2.append(f"{obj}_time")  # computation time
    columns_2.append(f"{obj}_im_en") # imbalance energy
res = pd.DataFrame(data=np.NaN, columns=columns_1 + columns_2, index=range(n))

# initialize data dictionary for algos:
data = {}
data['dt'] =  1/4  # sampling time (h)
data['objectives'] = cfg['objectives']

# generate random household selections: "growing village"
H_max = max(cfg['households list'])
data['village demands']   = []
data['village batteries'] = []
for sample in range(cfg['samples']):
    # household names:
    H_selection_ind = rng.choice(len(df_demands.columns), H_max)
    H_selection_names = df_demands.columns[H_selection_ind].to_list()
    data['village demands'].append(H_selection_names)
    # batteries:
    bat_sample = {}
    for key in bat.keys():
        bat_sample[key[:-6]] = rng.uniform(low=bat[key][0], high=bat[key][1], size=H_max).tolist()
    bat_sample['S_end_min'] = [cfg['S_end_min']]*H_max
    data['village batteries'].append(bat_sample)
cfg['village demands']   = data['village demands']
cfg['village batteries'] = data['village batteries']
    
# all the for-loops:
counter = 0
progress_counter = 0
progress_max = len(cfg['periods list'])*len(cfg['households list'])
P_max = max(cfg['periods list'])
break_through = {algo:[P_max + 1, H_max + 1] for algo in cfg['algos'].keys()}

# for periods in tqdm(cfg['periods list'], desc="periods", position=0):
for periods in cfg['periods list']:

    # select time slices: symmetric around 12:00 + x % phasing out
    noon_ind = int(12/data['dt'])
    time_ind_eval = np.arange(start=noon_ind - periods/2, stop=noon_ind + periods/2, 
                              step=1, dtype=int)
    fade_out = np.round(periods*cfg['fade out'])
    time_ind_all =  np.arange(start=noon_ind - periods/2, stop=noon_ind + periods/2 + fade_out, 
                              step=1, dtype=int)
    data['periods'] = periods
    
    # for households in tqdm(cfg['households list'], desc="households", position=0, leave=False):
    for households in cfg['households list']:

        # progress:
        progress = progress_counter/progress_max*100
        print(f"{progress = :.1f} % so far, now working in {periods} periods with {households} households ...")
        progress_counter += 1

        data['households'] = households
        
        # create multiprocessing pool if necessary:
        if cfg['multiprocessing processes'] > 1:
            pool = multiprocessing.Pool(processes=cfg['multiprocessing processes'])
        
        for day in cfg['days']:
            
            day_ts = pd.Timestamp(day)
            day_after_ts = day_ts + pd.Timedelta(1, unit='day')
            
            # select prices of this and following day:
            data['prices'] = df_prices.loc[day_ts:day_after_ts,].values.flatten()[time_ind_all]
            
            # apply algos:
            for algo_description, sub_module in cfg['algos'].items():
                if periods >= break_through[algo_description][0] and households >= break_through[algo_description][1]:
                    continue
                
                algo_function = getattr(lib, sub_module).algo
                
                data_list = []
                for sample in range(cfg['samples']):
                    
                    # select households' demands of this sample:
                    H_selection_names = data['village demands'][sample]
                    data['demands'] = df_demands.loc[day_ts:day_after_ts, 
                                                     H_selection_names[:households]].values[time_ind_all]

                    # select households batteries  of this sample:
                    bat_sample = data['village batteries'][sample].copy()
                    for key in bat_sample.keys():
                        bat_sample[key] = bat_sample[key][:households]
                    data['batteries'] = bat_sample
                    data['sample'] = sample
                
                    # append sample data to list:
                    data_list.append(data.copy())
                
                # compute results with algo-function:
                t0_algo_map = time.time()
                if cfg['multiprocessing processes'] > 1:
                    algo_res_list = pool.map(algo_function, data_list)
                else:
                    algo_res_list = list(map(algo_function, data_list))
                time_algo_map = time.time() - t0_algo_map
                if time_algo_map > cfg['breakthrough time']: # counter > 3
                    break_through[algo_description] = [periods, households] 
                    print(f"\n--- >>> BREAKTHROUGH for algo '{algo_description}' at {periods = } and {households = }! <<<<\n")
                
                for k in range(cfg['samples']):
                    algo_res = algo_res_list[k]
                    sample = algo_res['sample']
                    # insert results into dataframe:
                    res.loc[counter, columns_1] = [periods, households, day, sample, algo_description, algo_res['algo time']]
                    res.loc[counter, columns_2] = [algo_res[key] for key in columns_2]
                    counter += 1
                    
        # close multiprocessing pool:
        if cfg['multiprocessing processes'] > 1:
            pool.close()    

# breakthrough clean up:
res.dropna(axis=0, how='all', inplace=True)

# stopping time of ovaerall computations:
cfg['overall time'] = time.time() - t0
print(f"\n--- FINISHED ALL COMPUTATIONS after {cfg['overall time']/60.0:.2f} MINUTES. ---")

# export cfg and results with json and pickle:
if not os.path.exists(cfg['results path']):
    os.makedirs(cfg['results path'])
with open(cfg['results path'] + 'cfg.json', 'w') as outfile:
    json.dump(cfg, outfile, indent=2)
with open(cfg['results path'] + 'res.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"\n--- CFG AND RESULTS SAVED TO DIRECTORY {cfg['results dir']}. ---")
print(f"\n=== FINISHED ALL. ===")


# In[13]:


res #.head(50)


# In[ ]:




