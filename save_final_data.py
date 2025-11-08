import pandas as pd

results = [
    {'N': 100,  'epsilon_opt': 0.003518, 'd_PR': 3.05, 'd_MDS': 10, 'S_ratio': 14.213},
    {'N': 200,  'epsilon_opt': 0.002351, 'd_PR': 3.13, 'd_MDS': 10, 'S_ratio': 14.926},
    {'N': 500,  'epsilon_opt': 0.001307, 'd_PR': 3.14, 'd_MDS': 11, 'S_ratio': 15.458},
    {'N': 1000, 'epsilon_opt': 0.000809, 'd_PR': 3.19, 'd_MDS': 11, 'S_ratio': 16.083},
    {'N': 2000, 'epsilon_opt': 0.000465, 'd_PR': 3.20, 'd_MDS': 11, 'S_ratio': 16.511},
]

df = pd.DataFrame(results)
df.to_csv('results/data/final_results.csv', index=False)

print("✓ Data saved to: results/data/final_results.csv")
print("\nFinal dataset:")
print(df.to_string(index=False))