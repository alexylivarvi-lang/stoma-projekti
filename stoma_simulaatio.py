import numpy as np
from scipy.integrate import solve_ivp

# Fysikaaliset ja mekaaniset vakiot
R = 8.314        # Kaasuvakio J/(mol K)
T = 298.15       # Lämpötila (K)
F = 96485.0      # Faradayn vakio C/mol
z_K = 1.0        # K+ varaus

A_cell = 1e-9    # Sulusolun pinta-ala (m²)
V_0 = 4e-15      # Lepo-/minimitilavuus, jossa turgoripaine on nolla (m³)
epsilon = 25.0   # Soluseinän kimmomoduuli (MPa)
Cm = 0.1         # Kalvokapasitanssi (F/m²)
Lp = 2e-6        # Hydraulinen konduktiivisuus (m/(s MPa))
K_out = 10.0     # Ulkoinen K+ pitoisuus (mol/m3 eli mM)
Psi_w_out = -0.05 # Ulkoinen vesipotentiaali (MPa)

def stoma_ode_system(t, y, light_active, aba_level):
    Vm, n_K, n_Cl, n_mal, V = y
    V = max(V, V_0 * 0.99)
    
    # 1. Pitoisuudet ja osmootit (rajoitetaan fysiologiseen maksimiin)
    c_K = n_K / V
    c_Cl = n_Cl / V
    c_mal = n_mal / V
    c_muut = 30.0 * (V_0 / V) 
    Ci = c_K + c_Cl + c_mal + c_muut  
    
    # 2. Nernstin tasapainojännite ja konduktanssit
    c_K_safe = max(c_K, 1.0)
    E_K = (R * T / (z_K * F)) * np.log(K_out / c_K_safe)
    
    # Konduktanssit valo- ja pimeys-/ABA-tiloille
    g_max_in = 8.0 if light_active else 0.1
    g_max_out = 1.0 if light_active else 25.0  
    
    V_half_in, S_in = -0.15, 0.015
    V_half_out, S_out = -0.05, 0.02
    
    g_K_in = g_max_in / (1.0 + np.exp((Vm - V_half_in) / S_in))
    g_K_out = g_max_out / (1.0 + np.exp((V_half_out - Vm) / S_out))
    
    # Estetään K+ sisäänvirtaustulva, jos pitoisuus ylittää fysiologisen rajan (esim. 500 mM)
    if c_K > 500.0 and light_active:
        g_K_in = 0.0
        
    I_K = (g_K_in + g_K_out) * (Vm - E_K) 
    
    # 3. Protonipumppu voltage-saturaatiolla (ei tue hyperpolarisaatiota yli -0.17 V)
    max_pump = 2.0 if light_active else -0.3
    max_pump *= (1.0 + 3.0 * aba_level) if not light_active else 1.0
    
    # Pumppu hiipuu, kun Vm lähenee -170 mV (-0.17 V)
    pump_factor = max(0.0, 1.0 - (Vm / -0.17)) if Vm < 0 else 1.0
    I_H_pump = max_pump * pump_factor
    
    anion_activation = aba_level if not light_active else 0.01
    I_Cl = 2.0 * anion_activation * (Vm - (-0.03))
    I_mal = 2.0 * anion_activation * (Vm - (-0.02))
    I_leak = 0.05 * Vm  # Vahvempi vuotovirta stabiloimaan potentiaalia
    
    I_tot = I_H_pump + I_K + I_Cl + I_mal + I_leak
    
    # 4. Osmoottinen potentiaali ja turgoripaine
    i_osmotic = 1.0
    Psi_s = - (i_osmotic * R * T * Ci) / 1e6  # MPa
    Psi_p = max(0.0, epsilon * (V - V_0) / V_0) # MPa
    
    J_v = Lp * (Psi_w_out - (Psi_s + Psi_p))
    
    # 5. Derivaatat
    dVm_dt = -I_tot / Cm
    dn_K_dt = - (A_cell / F) * I_K
    dn_Cl_dt = + (A_cell / F) * I_Cl
    
    # Malaatin synteesi valossa, nopea hajoaminen pimeässä/ABA:ssa
    r_syn = 1.0e-16 if light_active else 0.0  
    r_deg = 1.5e-15 if not light_active else 0.0  
    dn_mal_dt = + (A_cell / (2.0 * F)) * I_mal + r_syn - r_deg
    
    dV_dt = A_cell * J_v
    
    return [dVm_dt, dn_K_dt, dn_Cl_dt, dn_mal_dt, dV_dt]

if __name__ == "__main__":
    # Alkutila: realistiset lähtöarvot
    y0 = [
        -0.120,                # Vm = -120 mV
        40.0 * 1e-3 * V_0,     # n_K 
        20.0 * 1e-3 * V_0,     # n_Cl 
        20.0 * 1e-3 * V_0,     # n_mal 
        V_0                    
    ]
    
    # Simulaatio: Valo (0–30 min) -> Pimeys + ABA (30–60 min)
    sol1 = solve_ivp(lambda t, y: stoma_ode_system(t, y, True, 0.0), (0, 1800), y0, t_eval=np.linspace(0, 1800, 15), method='LSODA')
    sol2 = solve_ivp(lambda t, y: stoma_ode_system(t, y, False, 1.0), (1800, 3600), sol1.y[:, -1], t_eval=np.linspace(1800, 3600, 15), method='LSODA')
    
    times = np.concatenate((sol1.t, sol2.t)) / 60.0
    states = np.concatenate((sol1.y, sol2.y), axis=1).T
    
    Vm_arr = states[:, 0] * 1000.0 # mV
    n_K_arr = states[:, 1]
    V_arr = states[:, 4]
    
    c_K_arr = n_K_arr / V_arr
    Psi_p_arr = np.maximum(0.0, epsilon * (V_arr - V_0) / V_0)
    apertura_arr = 10.0 / (1.0 + np.exp(-5.0 * (Psi_p_arr - 1.0)))
    
    csv_filename = "stoma_simulaatio_fysiologinen.csv"
    header = "Aika_min,Kalvojannite_mV,K_pitoisuus_mM,Tilavuus_m3,Turgoripaine_MPa,Apertura_um"
    data_to_save = np.column_stack((times, Vm_arr, c_K_arr, V_arr, Psi_p_arr, apertura_arr))
    np.savetxt(csv_filename, data_to_save, delimiter=",", header=header, comments="")
    
    print(f"{'Aika (min)':<10} | {'Vm (mV)':<10} | {'[K+] (mM)':<12} | {'Turgor (MPa)':<14} | {'Apertura (µm)'}")
    print("-" * 65)
    for t, vm, ck, psi_p, ap in zip(times, Vm_arr, c_K_arr, Psi_p_arr, apertura_arr):
        print(f"{t:<10.1f} | {vm:<10.1f} | {ck:<12.1f} | {psi_p:<14.2f} | {ap:.2f}")
        
    print(f"\nFysiologisesti rajattu simulaatio valmis! Tulokset tallennettiin tiedostoon: {csv_filename}")