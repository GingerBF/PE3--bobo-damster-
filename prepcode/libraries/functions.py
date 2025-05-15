import numpy as np
from functools import partial

# A sample list of labels
function_metadata = {
    "RC_C": {"type": "low-pass", "description": "1st order RC circuit measured over C"},
    "RCRC_CC": {"type": "low-pass", "description": "2nd order RC circuit measured over C"},
    "RC_R": {"type": "high-pass", "description": "1st order RC circuit measured over C"},
    "RCRC_RR": {"type": "high-pass", "description": "2nd order RC measured over C"},
    "RLC_sR": {"type": "", "description": "Series RLC circuit measured over R"},
    "RLC_sL": {"type": "", "description": "Series RLC circuit measured over L"},
    "RLC_sC": {"type": "low-pass", "description": "Series RLC circuit measured over C"},
    "RLC_sRL": {"type": "", "description": "Series RLC circuit measured over R and L"},
    "RLC_sRC": {"type": "", "description": "Series RLC circuit measured over R and C"},
    "RLC_sCL": {"type": "", "description": "Series RLC circuit measured over L and C"},
    "RLC_pR": {"type": "", "description": "Parallel RLC circuit measured over R"},
    "RLC_pL": {"type": "", "description": "Parallel RLC circuit measured over L"},
    "RLC_pC": {"type": "", "description": "Parallel RLC circuit measured over C"},
    "RLC_pRL": {"type": "", "description": "Parallel RLC circuit measured over R and L"},
    "RLC_pRC": {"type": "", "description": "Parallel RLC circuit measured over R and C"},
    "RLC_pCL": {"type": "", "description": "Parallel RLC circuit measured over L and C"},
    "RL_R": {"type": "", "description": "RL circuit measured over R"},
    "RL_L": {"type": "", "description": "RL circuit measured over L"},
    "CL_C": {"type": "", "description": "CL circuit measured over C"},
    "CL_L": {"type": "", "description": "Cl circuit measured over L"},
}

# A helper function for calling transfer functions
def call_H_function(func, f, **kwargs):
    """Call a transfer function with given arguments."""
    return func(f, **kwargs)

def RC_C(f, R, L, C):
    RC = R * C
    return 1 / (1 + 2 * 1j * np.pi * f * RC)

def RC_R(f, R, L, C):
    RC = R * C
    return 2 * np.pi * f * RC * 1j / (1 + 2 * 1j * np.pi * f * RC)

def RCRC_CC(f, R, L, C):
    omega = 2 * np.pi * f
    s = omega * 1j
    numerator = 1
    denominator = (
        (C * R * s + 1)**2
    )
    return numerator / denominator

def RCRC_RR(f, R, L, C):
    omega = 2 * np.pi * f
    s = omega * 1j
    numerator = C**2 * R**2 * s**2
    denominator = (
        (C * R * s + 1)**2
    )
    return numerator / denominator
       
def RCRC_RC(f, R, R2, C, C2):
    omega = 2 * np.pi * f
    s = omega * 1j
    numerator = R2 * C2 * s
    denominator = (
        (R * C * s + 1) * (R2 * C2 * s + 1)
    )
    return numerator / denominator

def RLC_sR(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = (R) * s * C
    denominator = s**2 * L * C + s * L * C + 1
    return numerator / denominator
    
def RLC_sL(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = (s * L) * s * C
    denominator = s**2 * L * C + s * L * C + 1
    return numerator / denominator

def RLC_sC(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator =  1
    denominator = s**2 * L * C + s * R * C + 1
    return numerator / denominator

def RLC_sRL(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = (R + s * L) * s * C
    denominator = s**2 * L * C + s * L * C + 1
    return numerator / denominator

def RLC_sRC(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = (R) * s * C + 1
    denominator = s**2 * L * C + s * L * C + 1
    return numerator / denominator

def RLC_sLC(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = s**2 * L * C + 1
    denominator = s**2 * L * C + s * R * C + 1
    return numerator / denominator

def RLC_pR(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = R + R * L * C * s**2
    denominator = R + C * L * R * s**2 + L * s
    return numerator / denominator
    
def RLC_pL(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = C * L * R * s**2 + L * s
    denominator = R + C * L * R * s**2 + L * s
    return numerator / denominator
    
def RLC_pC(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = L * s + R
    denominator = R + C * L * R * s**2 + L * s
    return numerator / denominator
    
def RLC_pRL(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = R
    denominator = R + C * L * R * s**2 + L * s
    return numerator / denominator
    
def RLC_pRC(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = C * L * R * s**2
    denominator = R + C * L * R * s**2 + L * s
    return numerator / denominator
    
def RLC_pLC(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = L * s
    denominator = R + C * L * R * s**2 + L * s
    return numerator / denominator

def RL_R(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = R
    denominator = R + L * s
    return numerator / denominator

def RL_L(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = L * s
    denominator = R + L * s
    return numerator / denominator

def CL_C(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = 1
    denominator = C*L*s**2 + 1
    return numerator / denominator
    
def CL_L(f, R, L, C):
    s = 1j * 2 * np.pi * f
    numerator = C*L*s**2
    denominator = C*L*s**2 + 1
    return numerator / denominator

def RLC_pR_RL(f, R, R2, L, C):
    s = 1j * 2 * np.pi * f
    numerator = R + C * L * R * s**2 + C * R * R2 * s
    denominator = R + R2 + C * L * R * s**2 + L * s + C * R * R2 * s
    return numerator / denominator
