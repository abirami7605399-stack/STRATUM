"""Global configuration for the STRATUM experimental framework.

All paths, signal-processing constants, corpus-construction parameters and
default optimisation settings are collected here so that a single edit
propagates through preprocessing, training and evaluation.
"""

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
MITDB_DIR = os.environ.get("MITDB_DIR", "data/mitdb")
MHEALTH_DIR = os.environ.get("MHEALTH_DIR", "data/mhealth")
CACHE_DIR = os.environ.get("STRATUM_CACHE", "cache")
RESULT_DIR = os.environ.get("STRATUM_RESULTS", "results")
FIGURE_DIR = os.environ.get("STRATUM_FIGURES", "figures")

for _d in (CACHE_DIR, RESULT_DIR, FIGURE_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Signal parameters
# --------------------------------------------------------------------------
FS = 50                      # common working sampling rate (Hz)
FS_MITDB = 360               # native MIT-BIH sampling rate (Hz)
BEAT_WIN_S = 2.0             # beat segment duration (s)
BEAT_LEN = int(BEAT_WIN_S * FS)          # 100 samples
CTX_WIN_S = 10.0             # inertial context window duration (s)
CTX_LEN = int(CTX_WIN_S * FS)            # 500 samples
CTX_DS = 5                   # decimation applied to the context window
CTX_LEN_DS = CTX_LEN // CTX_DS           # 100 samples per inertial channel
N_IMU_CH = 9                 # chest, ankle and lower-arm tri-axial acceleration

ECG_BAND = (0.5, 20.0)       # analysis band for the ECG branch (Hz)
QRS_BAND = (5.0, 15.0)       # detection band for R-peak localisation (Hz)

# --------------------------------------------------------------------------
# Label spaces
# --------------------------------------------------------------------------
AAMI_CLASSES = ["N", "S", "V"]
N_RHYTHM = len(AAMI_CLASSES)

# Fusion and unclassifiable beats are left out: their prevalence is below one
# percent and, at the 50 Hz working rate of a consumer-grade wearable, they are
# not separable from the classes they fuse.
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
}

ACTIVITY_NAMES = [
    "standing", "sitting", "lying", "walking", "stairs", "waist-bend",
    "arm-elevation", "knee-bend", "cycling", "jogging", "running", "jumping",
]
N_ACTIVITY = len(ACTIVITY_NAMES)

# Metabolic-intensity grouping used for the motion-stratified analysis.
# 0 = sedentary, 1 = light/moderate, 2 = vigorous.
ACTIVITY_INTENSITY = [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2]

# --------------------------------------------------------------------------
# Patient partitions (inter-patient protocol, paced records excluded)
# --------------------------------------------------------------------------
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
       124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
       212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
PACED = [102, 104, 107, 217]

MHEALTH_SUBJECTS = list(range(1, 11))

# --------------------------------------------------------------------------
# Corpus construction
# --------------------------------------------------------------------------
GRAFT_RATE = 0.45            # fraction of host beats carrying donor morphology
GRAFT_RATE_EVAL = 0.70       # higher rate on evaluation hosts for stable statistics
KAPPA_TRAIN = 0.80           # activity-ectopy spurious coupling in training
KAPPA_EVAL = 0.0             # coupling removed in the primary evaluation set
KAPPA_ADVERSE = -0.80        # coupling reversed in the adversarial evaluation set

# --------------------------------------------------------------------------
# Federation topology
# --------------------------------------------------------------------------
N_EDGE_CELLS = 4
LOCAL_EPOCHS = 1             # tau_1: device steps between edge aggregations
EDGE_PERIOD = 3              # tau_2: edge rounds between cloud aggregations
N_ROUNDS = 60                # cloud rounds

# --------------------------------------------------------------------------
# Optimisation defaults
# --------------------------------------------------------------------------
LR = 3e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
EMB_DIM = 64
SEEDS = [0, 1, 2, 3, 4]

# Objective weights (lambda_a, lambda_v, lambda_c, lambda_f, lambda_r)
W_ACTIVITY = 0.50
W_VITAL = 0.35
W_ADVERSARY = 0.25
W_COUNTERFACTUAL = 0.20
W_PROX = 1e-3

# Averaging window for the reported heart rate (s)
HR_AVG_S = 10.0
