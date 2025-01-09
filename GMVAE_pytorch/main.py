import os
import json
import torch
import pickle
import argparse
import numpy as np
import mdtraj as md

from os.path import join
from model import GMVAE
from train import train
from eval import evaluate
from torch.optim import Adam
from utils import TimeLaggedDataset, analyze_gmvae_clustering, print_traj_properties
from torch.utils.data import DataLoader
from weighted_rmsd_fitting import WeightedRMSDFit
from sklearn.preprocessing import StandardScaler, MinMaxScaler

argparser = argparse.ArgumentParser()
# Data
argparser.add_argument('--pdb_path', type=str) 
argparser.add_argument('--xtc_path', type=str)
argparser.add_argument('--condition', type=str, help='i.e. WT_apo_ChainsA_CA')
argparser.add_argument('--norm' , type=str, default='minmax', help='standard or minmax')
argparser.add_argument('--partition', type=float, default=0.8)
argparser.add_argument('--stride', type=int, default=1, help='Frame stride for xtc file')
# Fitting
argparser.add_argument('--fit_traj', action='store_true', default=False)
argparser.add_argument('--ref_pdb', type=str, default=None)
argparser.add_argument('--sfactor', type=float, default=5.0)
# Training
argparser.add_argument('--hyperparameters', type=str, default='arquitecture1.json')
argparser.add_argument('--time_lag', type=int, default=None)
argparser.add_argument('--load_model', type=str, default=None)
argparser.add_argument('--train_model', action='store_true', default=False)
argparser.add_argument('--seed', type=int, default=42)
args = argparser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  
np.random.seed(args.seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Create output directory
hparams_filename = args.hyperparameters.split('/')[-1].split('.')[0]
if args.time_lag:
    condition_folder = join("Results", "TGMVAE", args.condition)
else:
    condition_folder = join("Results", "GMVAE", args.condition)
save_folder = join(condition_folder, hparams_filename)
os.makedirs(save_folder, exist_ok=True)

# Load data
# Weighted RMSD fit
if args.fit_traj:
    print("Fitting trajectory...")
    fitted_traj = WeightedRMSDFit(args.pdb_path, 
                                  args.xtc_path,  
                                  args.sfactor, 
                                  args.ref_pdb,
                                  args.stride)

    # Save fitted trajectory
    fitted_mdtraj = md.Trajectory(fitted_traj, md.load(args.pdb_path).topology)
    fitted_mdtraj[0].save(join(condition_folder,args.condition+'_fit.pdb'))
    fitted_mdtraj.save_xtc(join(condition_folder,args.condition+'_fit.xtc'))
    print("Fitted trajectory saved...\n")
else:
    fitted_mdtraj = md.load(args.xtc_path, top=args.pdb_path, stride=args.stride)
    fitted_traj = fitted_mdtraj.xyz

print_traj_properties(fitted_mdtraj)

# Flatten the coordinates
coords = fitted_traj.reshape(fitted_traj.shape[0], -1)

# Normalize data
if args.norm == 'standard':
    scaler = StandardScaler()
    coords = scaler.fit_transform(coords)
elif args.norm == 'minmax':
    scaler = MinMaxScaler(feature_range=(-1, 1))
    coords = scaler.fit_transform(coords)
else:
    pass

# Save Sklearn scaler
with open(join(save_folder, args.condition+'_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# Get hyperparameters
with open(args.hyperparameters, 'r') as f:
    hparams = json.load(f)
    print("Hyperparameters: ", hparams, "\n")

# Save Job Parameters
with open(join(save_folder, 'job_parameters.json'), 'w') as f:
    json.dump(hparams, f)
    json.dump(vars(args), f)

# Split data
n_train = int(args.partition * len(coords))
train_data = coords[:n_train]
valid_data = coords[n_train:]

# Time-lagged data
if args.time_lag:
    train_data = TimeLaggedDataset(train_data, args.time_lag)
    valid_data = TimeLaggedDataset(valid_data, args.time_lag)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=hparams["batch_size"], drop_last=True, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=hparams["batch_size"], drop_last=False, shuffle=False)
test_loader = DataLoader(coords, batch_size=hparams["batch_size"], drop_last=False, shuffle=False)

# Initialize model
model = GMVAE(k=hparams["k"], 
              n_x=coords.shape[1], 
              n_z=hparams["n_z"], 
              qy_dims=hparams["qy_dims"], 
              qz_dims=hparams["qz_dims"], 
              pz_dims=hparams["pz_dims"], 
              px_dims=hparams["px_dims"], 
              r_nent=hparams["r_nent"],
              use_batch_norm=hparams["use_batch_norm"])

if args.load_model:
    model.load_state_dict(torch.load(args.load_model, map_location=device))
    print("Model loaded...\n")

model.to(device)
print(model, "\n")

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=hparams["lr"])

# Train model
if args.train_model:
    model, training_losses = train(model, 
                                   train_loader, 
                                   valid_loader, 
                                   optimizer, 
                                   hparams["epochs"], 
                                   args.time_lag, 
                                   device)

    # Save model
    torch.save(model.state_dict(), join(save_folder, args.condition+'_model.pt'))
    print("Model saved...\n")

    # Save training losses
    with open(join(save_folder, 'training_losses.pkl'), 'wb') as f:
        pickle.dump(training_losses, f)
    

# Evaluate model
print("Evaluate...\n")
qy, qy_log, z_list, x_list = evaluate(model, test_loader, device)
analyze_gmvae_clustering(qy)

# Latent Space (z)
z = model.sum_aggregation(z_list, qy)

# Reconstruction (x')
x = model.sum_aggregation(x_list, qy)
x = scaler.inverse_transform(x)
x = np.reshape(x, (len(x), -1, 3))

# Save results
np.savez(join(save_folder, args.condition+'_clustering.npz'), qy=qy, z=z)
np.savez(join(save_folder, args.condition+'_results.npz'), qy=qy, qy_log=qy_log, z=z, x=x)

print("\nResults saved at: ", save_folder+"/")
print("\nDone!")
