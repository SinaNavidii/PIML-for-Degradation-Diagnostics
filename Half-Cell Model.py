import numpy as np
import scipy.io
import scipy.interpolate

def halfcell(DPs, file_path='halfcell.mat', Vexp_min=3.4, Vexp_max=4.075, data_no=500, Q_sim_range=(0.05, 100)):
    """
    Half-cell model function.

    Parameters:
    - DPs (list): List of parameters [m_P, m_N, d_P, d_N]
        where m_P (g) and m_N (g) are the masses, and d_P, d_N are slippage parameters.
    - file_path (str): Path to the half-cell data file.
    - Vexp_min (float): Minimum experimental voltage.
    - Vexp_max (float): Maximum experimental voltage.
    - data_no (int): Number of data points for output voltage-based simulation.
    - data_no1 (int): Number of data points for Q_sim and interpolations.
    - Q_sim_range (tuple): Range for simulated cell capacity as (min, max).

    Returns:
    - dQdV (np.ndarray): Differential capacity (dQ/dV).
    - Qf_sim_Vbased (np.ndarray): Simulated capacity based on voltage.
    """
    m_P, m_N, d_P, d_N = DPs  # Unpack parameters

    # Load half-cell data
    halfcell_data = scipy.io.loadmat(file_path)
    q_PE_hc = halfcell_data['halfcell'][0][0]
    V_PE_hc = halfcell_data['halfcell'][0][1]
    q_NE_hc = halfcell_data['halfcell'][0][2]
    V_NE_hc = halfcell_data['halfcell'][0][3]

    # Simulate cell capacity range
    Q_sim = np.linspace(Q_sim_range[0], Q_sim_range[1], data_no)

    # Interpolate half-cell data
    qPE_hc_sim = np.linspace(np.min(q_PE_hc), np.max(q_PE_hc), data_no)
    qNE_hc_sim = np.linspace(np.min(q_NE_hc), np.max(q_NE_hc), data_no)
    VPE_hc_sim = scipy.interpolate.PchipInterpolator(q_PE_hc.flatten(), V_PE_hc.flatten())(qPE_hc_sim)
    VNE_hc_sim = scipy.interpolate.PchipInterpolator(q_NE_hc.flatten(), V_NE_hc.flatten())(qNE_hc_sim)

    # Calculate adjusted capacities for PE and NE half-cells
    Q_PE_hc = qPE_hc_sim * m_P
    Q_NE_hc = qNE_hc_sim * m_N
    Q_NE_hc_ad = Q_NE_hc - d_N
    Q_PE_hc_ad = Q_PE_hc - d_P

    # Interpolate experimental data over Q_sim range
    VPE_sim = scipy.interpolate.interp1d(Q_PE_hc_ad, VPE_hc_sim, 'linear', fill_value="extrapolate")(Q_sim)
    VNE_sim = scipy.interpolate.interp1d(Q_NE_hc_ad, VNE_hc_sim, 'linear', fill_value="extrapolate")(Q_sim)
    
    # Voltage cutoff
    Vf_sim = VPE_sim - VNE_sim
    Q_sim = Q_sim[~np.isnan(Vf_sim)]
    Vf_sim = Vf_sim[~np.isnan(Vf_sim)]

    # Define voltage range indices
    Vmin_ID = np.argmin(np.abs(Vf_sim - Vexp_min))
    Vmax_ID = np.argmin(np.abs(Vf_sim - Vexp_max))
    Q_sim_1 = Q_sim[Vmin_ID:Vmax_ID] - Q_sim[Vmin_ID]
    Vf_sim_1 = Vf_sim[Vmin_ID:Vmax_ID]

    # Interpolate for finer resolution
    Q_sim_2 = np.linspace(Q_sim_1.min(), Q_sim_1.max(), data_no)
    Vf_sim_2 = scipy.interpolate.interp1d(Q_sim_1, Vf_sim_1, 'linear', fill_value="extrapolate")(Q_sim_2)

    # Calculate differential curves
    dVdQ_sim = np.diff(Vf_sim_2) / np.diff(Q_sim_2)
    dVdQ_sim = np.append(dVdQ_sim, dVdQ_sim[-1])  # Keep length consistent
    dQdV_sim = np.diff(Q_sim_2) / np.diff(Vf_sim_2)
    dQdV_sim = np.append(dQdV_sim, dQdV_sim[-1])

    # V-Q reconstructed 
    Vf_sim_Vbased = np.linspace(Vexp_min, Vexp_max, data_no)
    Qf_sim_Vbased = scipy.interpolate.interp1d(Vf_sim_2, Q_sim_2, 'linear', fill_value="extrapolate")(Vf_sim_Vbased)
    dQdV = scipy.interpolate.interp1d(Vf_sim_2, dQdV_sim, 'linear', fill_value="extrapolate")(Vf_sim_Vbased)

    return dQdV, Qf_sim_Vbased
