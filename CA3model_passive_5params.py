from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def generate_step_stim_s(I_max, hyperpol_t=500, dur=300, interpulse=100, dt=0.1):
    steps = np.array([-1, I_max, -1])
    durations = [hyperpol_t+interpulse, dur, interpulse+hyperpol_t+interpulse+dur+interpulse]
    stim_list = []
    for step_val, dur in zip(steps, durations):
        npts = int(round(dur / dt))
        stim_list.extend([step_val] * npts)
    stim = np.array(stim_list, dtype=float)
    t_stim = np.arange(0, len(stim) * dt, dt)
    return interp1d(t_stim, stim, kind='cubic', bounds_error=False, fill_value="extrapolate")

def generate_step_stim_d(I_max, hyperpol_t=500, dur=300, interpulse=100, dt=0.1):
    steps = np.array([-1, I_max, -1])
    durations = [hyperpol_t+interpulse+dur+interpulse+hyperpol_t+interpulse, dur, interpulse]
    stim_list = []
    for step_val, dur in zip(steps, durations):
        npts = int(round(dur / dt))
        stim_list.extend([step_val] * npts)
    stim = np.array(stim_list, dtype=float)
    t_stim = np.arange(0, len(stim) * dt, dt)
    return interp1d(t_stim, stim, kind='cubic', bounds_error=False, fill_value="extrapolate")

def CA3model_passive(vars,t, f_stim_ext_s, f_stim_ext_d, gL, VL, gc, pp, Cm):
    # injected current
    #Isom0=0.75 # -0.5
    Isom0=f_stim_ext_s(t) # microA/cm2
    Iden0=f_stim_ext_d(t)# -0.5 or 0

    Vs, Vd = vars

    ## the derivatives
    d_Vs=(-gL*(Vs-VL)+(gc/pp)*(Vd-Vs)+Isom0/pp)/Cm
    d_Vd=(-gL*(Vd-VL)+(gc*(Vs-Vd))/(1.0-pp)+Iden0/(1-pp))/Cm

    d_vars = [d_Vs, d_Vd]

    return d_vars

def CA3model_passive_solver(params, return_trace=False):

    f_stim_ext_s = generate_step_stim_s(-2)
    f_stim_ext_d = generate_step_stim_d(-2)

    gL  = params[0]
    VL = params[1]
    gc  = params[2]
    pp = params[3]
    Cm = params[4]

    Vs0=-61.39434150397358
    Vd0=-61.479722834294584

    init_var = [Vs0, Vd0]

    t = np.linspace(0, 2000, 20001)

    tcritical = [600, 900, 1600, 1900]
    sol = odeint(CA3model_passive, init_var, t, tcrit=tcritical, args=(f_stim_ext_s, f_stim_ext_d, gL, VL, gc, pp, Cm))
    #timeintervals 500, 100, 300, 100, 500, 100, 300, 100

    #traces
    #somatic injection
    t_cut1 = 5000
    t_cut2 = 10000

    s_sol1 = sol[t_cut1:t_cut2,0] - sol[t_cut1,0]
    d_sol1 = sol[t_cut1:t_cut2,1] - sol[t_cut1,1]

    #dendritic injection
    t_cut3 = 15000
    t_cut4 = 20000

    s_sol2 = sol[t_cut3:t_cut4,0] - sol[t_cut3,0]
    d_sol2 = sol[t_cut3:t_cut4,1] - sol[t_cut3,1]

    traces = [s_sol1, d_sol1, s_sol2, d_sol2]

    #points
    #somatic injection
    t_point1 = 6100
    t_point2 = 8900
    s_p1 = sol[t_point1, 0] - sol[t_cut1,0]
    s_p2 = sol[t_point2, 0] - sol[t_cut1,0]
    d_p1 = sol[t_point1, 1] - sol[t_cut1,1]
    d_p2 = sol[t_point2, 1] - sol[t_cut1,1]

    #dendritic injection
    t_point3 = 16100
    t_point4 = 18900
    s_p3 = sol[t_point3, 0] - sol[t_cut3,0]
    s_p4 = sol[t_point4, 0] - sol[t_cut3,0]
    d_p3 = sol[t_point3, 1] - sol[t_cut3,1]
    d_p4 = sol[t_point4, 1] - sol[t_cut3,1]

    #points = [s_p1, d_p1, s_p2, d_p2, s_p3, d_p3, s_p4, d_p4]
    rng_t = np.random.default_rng()
    scale = 0.5

    points = [rng_t.normal(s_p1, scale), rng_t.normal(d_p1, scale), rng_t.normal(s_p2, scale),
              rng_t.normal(d_p2, scale), rng_t.normal(s_p3, scale), rng_t.normal(d_p3, scale),
              rng_t.normal(s_p4, scale), rng_t.normal(d_p4, scale)]

    p1_time = t_point1-t_cut1
    p2_time = t_point2-t_cut1
    p3_time = t_point3-t_cut3
    p4_time = t_point4-t_cut3

    p_times = [p1_time, p2_time, p3_time, p4_time]

    if return_trace:
        return [points, p_times, traces]
    else:
        return points