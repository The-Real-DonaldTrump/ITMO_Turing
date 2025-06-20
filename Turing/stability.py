import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_derivatives(u, v, F, k):
    a = -v**2 - F
    b = -2 * u * v
    c = v**2
    d = 2 * u * v - (F + k)
    return a, b, c, d

def compute_conditions(a, b, c, d):
    trace = a + d
    delta = a * d - b * c
    if delta <= 0 or a == 0:
        return None, False
    try:
        sqrt_delta = np.sqrt(delta)
        sqrt_term = np.sqrt(delta - a * d)
        D_cr = (2 * delta - a * d + 2 * sqrt_delta * sqrt_term) / (a ** 2)
        is_instability = (
            (a + d < 0) and
            (delta > 0) and
            ((a < 0 and d > 0) or (a > 0 and d < 0)) and
            (2 > D_cr)
        )
        return D_cr, is_instability
    except:
        return None, False

def solve_stationary(F, k):
    A = -(F + k)
    B = F
    C = -F * (F + k)
    D = B**2 - 4*A*C
    if D < 0:
        return None
    sqrt_D = np.sqrt(D)
    v1 = (-B + sqrt_D) / (2 * A)
    v2 = (-B - sqrt_D) / (2 * A)
    results = []
    for v in [v1, v2]:
        if v > 0:
            u = (F + k) / v
            results.append((u, v))
    return results

# Диапазоны параметров
k_values = np.arange(0.001, 0.081, 0.001)
F_values = np.arange(0.001, 0.301, 0.001)

stable_points = []
unstable_points = []

for k in k_values:
    for F in F_values:
        pairs = solve_stationary(F, k)
        if pairs is None:
            continue
        for u, v in pairs:
            a, b, c, d = compute_derivatives(u, v, F, k)
            D_cr, instability = compute_conditions(a, b, c, d)
            if instability:
                unstable_points.append((k, F))
            else:
                stable_points.append((k, F))

# Создание DataFrame только с F и k
df_stable = pd.DataFrame(stable_points, columns=['k', 'F'])
df_unstable = pd.DataFrame(unstable_points, columns=['k', 'F'])

# Сохранение CSV
df_stable.to_csv("stable_cases.csv", index=False)
df_unstable.to_csv("unstable_cases.csv", index=False)

# Построение графика
plt.figure(figsize=(8, 6))
plt.scatter(df_stable['k'], df_stable['F'], color='green', s=10, label='устойчивые')
plt.scatter(df_unstable['k'], df_unstable['F'], color='red', s=10, label='неустойчивые')
plt.xlabel('k')
plt.ylabel('F')
plt.title('Область устойчивости и неустойчивости на плоскости (k, F)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
