import numpy as np
import pandas as pd
from scipy.optimize import minimize

def process_data(file_path, a=1, b=10):
    df = pd.read_csv(file_path, index_col=0)
    
    # Group by 'id' and check for missing values
    missing_values = df.groupby('id').apply(lambda x: x.isna().any(axis=None))
    
    # Get indices where there are no missing values
    idx = (missing_values[missing_values == False].index).to_list()
    
    # Select rows with complete data 
    clean = df.loc[df['id'].isin(idx)]
    
    # Calculate 'last_reward'
    clean['last_reward'] = clean['cumulative_reward'] - clean['reward']
    
    # Normalize 'actual_arousal'
    min_val_arousal = clean['actual_arousal'].min()
    max_val_arousal = clean['actual_arousal'].max()
    clean['normalized_actual_arousal'] = a + (clean['actual_arousal'] - min_val_arousal) / (max_val_arousal - min_val_arousal) * (b - a)
    
    # Normalize 'fairness_event'
    clean['normalized_fairness_event'] = (clean['fairness_event'] - clean['fairness_event'].min()) / (clean['fairness_event'].max() - clean['fairness_event'].min())
    
    # Normalize 'last_reward'
    clean['normalized_last_reward'] = (clean['last_reward'] - clean['last_reward'].min()) / (clean['last_reward'].max() - clean['last_reward'].min())
    
    # Normalize 'actual_valence'
    min_val_valence = clean['actual_valence'].min()
    max_val_valence = clean['actual_valence'].max()
    clean['normalized_actual_valence'] = a + (clean['actual_valence'] - min_val_valence) / (max_val_valence - min_val_valence) * (b - a)
    
    # Initialize 'ini' dictionary
    ini = {i: 1 for i in idx}
    
    return clean, ini


def BERM(df, params, T=True):
    eps = 1e-6
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    beta1, beta2, gamma = params
    total_loss = 0
    belief = {}

    for idv in df['id'].unique():
        belief[idv] = np.zeros(20)  
        sub_df = df[df['id'] == idv].reset_index(drop=True)  
        for index, ser in sub_df.iterrows():
            
            if index == 0:
                global ini
                y = ini[idv]

            diff = -(1- beta1 * (y * ser['normalized_fairness_event']) -  beta2 * (ser['normalized_last_reward']) ) 
            if T:
                p = sigmoid((diff) / (eps + ser['normalized_actual_arousal'] + ser['normalized_actual_valence']) )
            else:
                p = sigmoid(diff)
            loss = - (ser['choice'] * np.log(p + eps) + (1 - ser['choice']) * np.log(1 - p + eps))  
            
            y = np.log(max(eps, np.exp(y) + gamma *  (ser['choice'] - p))) 

            if pd.isna(loss):
                print(ser)
            total_loss += loss  
            belief[idv][index] = y  

    return total_loss, belief  


def optimize_function(initial_guess, source, maxiter=1000):
    def objective(x):
        return BERM(clean.loc[clean['type'] == source], x)[0]
    
    constraints = ({'type': 'ineq', 'fun': lambda x: x})
    result = minimize(objective, initial_guess, constraints=constraints, options={'maxiter': maxiter})

    return result.x, result.fun

def calculate_mean_y(belief, idv_list):
    mean_y = np.zeros(20).astype('float32')  
    count = 0  

    for idv in idv_list:
        
        if idv in belief and not np.any(pd.isna(belief[idv])):
            mean_y += belief[idv]  
            count += 1  
           
    if count > 0:
        mean_y /= count  

    return mean_y  

def run_epochs(initial_guess, num_epochs=10):
    params = initial_guess

    for epoch in range(num_epochs):
        params, min_loss = optimize_function(params, s)
        print(f'Epoch {epoch+1}: loss = {min_loss}, params = {params}')
        
    return params

if __name__ == "__main__":
    N = 100
    M = 500
    clean, ini = process_data('data.csv') 
    types = ['human', 'gpt3.5', 'gpt4', 'gpt4o']
    beliefs = {}
    beliefs_mean = {}
    for s in types:
        param, _ = optimize_function((0.1, 0.1, 0.1), s)
        final_params = run_epochs(param, num_epochs=10)
        beliefs[f'{s}_belief'] = BERM(clean.loc[clean['type'] == s], param)[1]
        beliefs_mean[f'{s}_belief'] = calculate_mean_y(BERM(clean.loc[clean['type'] == s], final_params)[1], clean.loc[clean['type'] == s, 'id'].unique())

