import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# ======
# DATOS#
# ======
agbx = 89.0         
ngbx = 1.0          
R = 27.0            
beta = 0.0    
Wtur_rpm = 22.0     
rho = 1.3 
c1, c2, c3, c4, c5, c6 = 0.5, 1.0, 0.022, 2.0, 5.6, 0.17
Pbase = 0.5e6       
P_polos = 4.0
f = 60.0            
H = 0.2             
rs = 0.0059
rr = 0.0208
Xls = 0.0754
Xlr = 0.0326
Xm = 2.5326
Vqs = 1.0
Vds = 0.0

# =========
# CALCULOS#
# =========
pi = np.pi
w_s = (2 * pi * f) / (P_polos / 2.0)  
Wt_rad = Wtur_rpm * (2 * pi / 60.0)    

Xss = Xls + Xm
Xrr = Xlr + Xm
X_I = Xss - (Xm**2 / Xrr)
T0 = Xrr / (w_s * rr)
V_viento = 9.5    

landa = 2.237 * (V_viento / Wt_rad)
Cp = c1 * (c2 * landa - c3 * (beta**c4) - c5) * np.exp(-c6 * landa)
Area = pi * (R**2)
P_turbina = 0.5 * rho * Area * (V_viento**3) * Cp
T_tur_Nm = P_turbina / Wt_rad
T_gen_mec_Nm = (ngbx * T_tur_Nm) / agbx
T_base = Pbase / w_s
Tg_pu = T_gen_mec_Nm / T_base

print(f"--- DATOS CALCULADOS ---")
print(f"Viento: {V_viento} m/s")
print(f"Par Mecánico (pu): {Tg_pu:.4f}")

def ecuaciones(t, y):
    E, delta, wg = y
    if abs(E) < 1e-6: E = 1e-6

    dE = -(Xss * E) / (X_I * T0) + \
         ((Xss - X_I) / (X_I * T0)) * (Vqs * np.cos(delta) - Vds * np.sin(delta))
    
    ddelta = wg - w_s - \
             ((Xss - X_I) / (X_I * T0 * E)) * (Vqs * np.sin(delta) + Vds * np.cos(delta))
             
    dwg = (Tg_pu * w_s) / (2 * H) - \
          ((E * w_s) / (X_I * 2 * H)) * (Vds * np.cos(delta) + Vqs * np.sin(delta))
          
    return np.array([dE, ddelta, dwg])

# ========================
# METODO DE ADAMS-MOULTON+ADAMS-BASHFORTH#
# =========================
#Usamos prediccion de Adams-Bashforth y correcion de Adams-Moulton
#donde el primer dato de predicon se obtiene con euler
#y de hay en adelante con Adams-Bashforth de segundo orden
def ADAMS(h, t_max, y0):
    N = int(t_max / h)
    t = np.linspace(0, t_max, N+1)
    y = np.zeros((N+1, 3))
    y[0] = y0
    f0 = ecuaciones(t[0], y[0])
    y[1] = y[0] + h * f0
    for i in range(1, N):
        fn   = ecuaciones(t[i],   y[i])
        fn_1 = ecuaciones(t[i-1], y[i-1])
        y_pred = y[i] + (h/2)*(3*fn - fn_1)
        y_corr = y_pred
        for _ in range(3):
            fn1 = ecuaciones(t[i+1], y_corr)
            y_corr = y[i] + (h/12)*(5*fn1 + 8*fn - fn_1)
        y[i+1] = y_corr
    return t, y


# =======================
# CONDICIONES DEL MODELO#
# =======================

h_step = 0.0005
t_total = 2.0
y_init = np.array([0.1, 0.0, Wt_rad]) 

tiempo, resultados = ADAMS(h_step, t_total, y_init)

E_res = resultados[:, 0]
d_res = resultados[:, 1]
w_res = resultados[:, 2]
fig = make_subplots(rows=3, cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=("Voltaje  (E)", "Ángulo (Delta)", "Velocidad del Generador (Wg)"))

fig.add_trace(go.Scatter(x=tiempo, y=E_res, name="E (pu)", line=dict(color='blue')), row=1, col=1)

fig.add_trace(go.Scatter(x=tiempo, y=d_res, name="Delta (rad)", line=dict(color='green')), row=2, col=1)

fig.add_trace(go.Scatter(x=tiempo, y=w_res, name="Wg (rad/s)", line=dict(color='red')), row=3, col=1)



fig.update_layout(
    title_text=f"Simulación Del Generador Eólico A Una Velocidad De (Viento De ={V_viento} m/s)",
    height=900,  
    hovermode="x unified", 
    template="plotly_white"
)
fig.update_yaxes(title_text="p.u.", row=1, col=1)
fig.update_yaxes(title_text="radianes", row=2, col=1)
fig.update_yaxes(title_text="rad/s", row=3, col=1)
fig.update_xaxes(title_text="Tiempo (s)", row=3, col=1)

fig.show()