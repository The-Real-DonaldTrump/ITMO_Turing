import numpy as np
from scipy.optimize import fsolve
import pandas as pd
import matplotlib.pyplot as plt



def compute_derivatives(u, v, F, k):
    a = -v**2 - F             # ∂f1/∂u
    b = -2 * u * v            # ∂f1/∂v
    c = v**2                  # ∂f2/∂u
    d = 2 * u * v - (F + k)   # ∂f2/∂v
    return a, b, c, d

def compute_conditions(a, b, c, d):
    trace = a + d
    delta = a * d - b * c
    if delta <= 0 or a == 0:
        return trace, delta, None, False
    try:
        sqrt_delta = np.sqrt(delta)
        sqrt_term = np.sqrt(delta - a * d)
        D_cr = (2 * delta - a * d + 2 * sqrt_delta * sqrt_term) / (a ** 2)
        is_instability = (
                (a + d < 0) and
                (delta > 0) and
                ((a < 0 and d > 0) or (a > 0 and d < 0)) and
                (2 < D_cr)  # D (например, 2) должно быть больше D_cr
        )

        return trace, delta, D_cr, is_instability
    except:
        return trace, delta, None, False

results = []

k_values = np.arange(0.001, 0.101, 0.001)
F_values = np.arange(0.001, 0.101, 0.001)

for k in k_values:
    for F in F_values:
        def equation(v_array):
            v = v_array[0]
            if v == 0:
                return [1e6]
            u = (F + k) / v
            return [-u * v**2 + F * (1 - u)]

        try:
            v_initial = [0.1]
            v_sol = fsolve(equation, v_initial)[0]
            if np.abs(v_sol) > 1e-6:
                v = v_sol
                u = (F + k) / v
                a, b, c, d = compute_derivatives(u, v, F, k)
                trace, delta, D_cr, instability = compute_conditions(a, b, c, d)
                results.append((k, F, u, v, a, b, c, d, trace, delta, D_cr, instability))
        except Exception as e:
            continue

# Сохраняем таблицу
df = pd.DataFrame(results, columns=[
    'k', 'F', 'u', 'v', 'a', 'b', 'c', 'd',
    'a_plus_d', 'delta', 'D_cr', 'instability'
])

# Показываем только те строки, где выполняется неустойчивость
unstable_df = df[df['instability'] == True]
print(unstable_df.head())
unstable_df.to_csv("unstable_cases.csv", index=False)



# Строим график устойчивых случаев (instability == False)
stable_df = df[df['instability'] == False]

plt.figure(figsize=(8, 6))
plt.scatter(stable_df['k'], stable_df['F'], color='green', s=10, label='устойчивые')
plt.xlabel('k')
plt.ylabel('F')
plt.title('Область устойчивости на плоскости (k, F)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()