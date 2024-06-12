import numpy as np
import streamlit as st
from scipy.stats import norm
import matplotlib.pyplot as plt

# Function to calculate power using the z-test for proportions
def calculate_power(alpha, prop1, prop2, n1, n2, tail):
    # Calculate the pooled proportion
    pooled_prop = (prop1 * n1 + prop2 * n2) / (n1 + n2)
    # Calculate the standard error
    se = np.sqrt(pooled_prop * (1 - pooled_prop) * (1 / n1 + 1 / n2))
    # Calculate the effect size
    effect_size = (prop2 - prop1) / se
    # Determine the z critical value
    if tail == 'one':
        z_critical = norm.ppf(1 - alpha)
    else:
        z_critical = norm.ppf(1 - alpha / 2)
    # Calculate power
    if tail == 'one':
        power = 1 - norm.cdf(z_critical - effect_size)
    else:
        power = 1 - norm.cdf(z_critical - effect_size) + norm.cdf(-z_critical - effect_size)
    return z_critical, power, pooled_prop, effect_size

# Define the Streamlit app
def main():
    st.title("Post hoc power analysis")

    # Ask for user input with higher precision
    alpha = st.number_input("Alpha (α err prob)", min_value=0.0, max_value=1.0, value=0.05, format="%.5f")
    prop1 = st.number_input("Proportion p1", min_value=0.0, max_value=1.0, value=0.021, format="%.5f")
    prop2 = st.number_input("Proportion p2", min_value=0.0, max_value=1.0, value=0.031, format="%.5f")
    n1 = st.number_input("Sample size group 1", min_value=1, value=255)
    n2 = st.number_input("Sample size group 2", min_value=1, value=732)
    tail = st.selectbox("Tail(s)", ["one", "two"])

    # Calculate power and critical z
    z_critical, power, pooled_prop, effect_size = calculate_power(alpha, prop1, prop2, n1, n2, tail)

    st.write(f"Critical z: {z_critical:.5f}")
    st.write(f"Power (1-β err prob): {power:.5f}")

    # Plotting the power analysis
    x = np.linspace(-3, 3, 1000)
    y1 = norm.pdf(x)  # Standard normal distribution
    y2 = norm.pdf(x - effect_size)  # Shifted normal distribution

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label='H0', color='red')
    plt.plot(x, y2, label='H1', color='blue', linestyle='dashed')
    if tail == 'one':
        plt.fill_between(x, y1, where=(x > z_critical), color='red', alpha=0.3, label='α')
        plt.fill_between(x, y2, where=(x < z_critical), color='blue', alpha=0.3, label='β')
    else:
        plt.fill_between(x, y1, where=(x > z_critical) | (x < -z_critical), color='red', alpha=0.3, label='α')
        plt.fill_between(x, y2, where=(x < z_critical) & (x > -z_critical), color='blue', alpha=0.3, label='β')
    plt.axvline(z_critical, color='green', linestyle='dotted', label='critical z')
    if tail == 'two':
        plt.axvline(-z_critical, color='green', linestyle='dotted')
    plt.title('Power Analysis')
    plt.xlabel('z')
    plt.ylabel('Density')
    plt.legend()

    st.pyplot(plt)

if __name__ == "__main__":
    main()
