import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

def chi_squared(null, alt, train_index, train_cols):
    alpha = 0.05
    observed = pd.crosstab(train_index, train_cols)
    display(observed)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    if p < alpha:
        print("Reject the null hypothesis that", null)
        print("Sufficient evidence to move forward understanding that", alt)
    else:
        print("Fail to reject the null")
        print("Insufficient evidence to reject the null")
    print(' ')
    print(f'{p = }')