"""
Тестовый скрипт для проверки окружения
"""

import sys
import numpy as np
import scipy
import matplotlib
import pandas as pd
import tqdm

print("=" * 60)
print("ENVIRONMENT TEST")
print("=" * 60)

# Версии
print(f"\nPython:      {sys.version}")
print(f"NumPy:       {np.__version__}")
print(f"SciPy:       {scipy.__version__}")
print(f"Matplotlib:  {matplotlib.__version__}")
print(f"Pandas:      {pd.__version__}")
print(f"tqdm:        {tqdm.__version__}")

# Функциональный тест
print("\n" + "=" * 60)
print("FUNCTIONAL TEST")
print("=" * 60)

# 1. NumPy
print("\n[1/5] NumPy matrix operations...", end=" ")
A = np.random.randn(100, 100)
B = A @ A.T
eigenvalues = np.linalg.eigvalsh(B)
print("✓")

# 2. SciPy optimization
print("[2/5] SciPy optimization...", end=" ")
from scipy.optimize import minimize_scalar
result = minimize_scalar(lambda x: (x - 3.5)**2, bounds=(0, 10), method='bounded')
assert abs(result.x - 3.5) < 1e-6
print("✓")

# 3. Matplotlib
print("[3/5] Matplotlib plotting...", end=" ")
import matplotlib.pyplot as plt
plt.ioff()
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
plt.close()
print("✓")

# 4. Pandas
print("[4/5] Pandas DataFrame...", end=" ")
df = pd.DataFrame({'x': np.arange(10), 'y': np.random.rand(10)})
assert len(df) == 10
print("✓")

# 5. tqdm
print("[5/5] tqdm progress bar...", end=" ")
from tqdm import tqdm as tqdm_bar
import time
for i in tqdm_bar(range(5), desc="Test", leave=False):
    time.sleep(0.01)
print("✓")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nEnvironment is ready for holographic_dimension project!")