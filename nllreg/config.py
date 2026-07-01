"""Global configuration and paths for the NLLReg experiments."""
import os
import torch

# Repository layout ---------------------------------------------------------- #
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PKG_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")
PLOTS_DIR = os.path.join(REPO_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Problem / model constants -------------------------------------------------- #
EMB_DIM = 16
N_CLASSES = 10
HOLDOUT = 4                      # digit held out of the embedding in the holdout study

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_path(name):
    """Absolute path for a figure inside the plots/ directory."""
    return os.path.join(PLOTS_DIR, name)
