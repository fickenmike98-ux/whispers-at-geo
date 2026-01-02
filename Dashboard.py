import matplotlib
matplotlib.use('Qt5Agg') # Change from TkAgg to Qt5Agg
import matplotlib.pyplot as plt
import seaborn as sns

def plot_strategic_risk(lead_lag_results):
    """
    lead_lag_results: A dict or DataFrame containing {Sat_ID: Lead_Days}
    """
    plt.figure(figsize=(12, 6))
    # Rank satellites by their 'Lead' capability
    sns.barplot(x='Sat_ID', y='Lead_Days', data=lead_lag_results, palette='magma')
    plt.title("SDA STRATEGIC LEAD TIME: DAYS BEFORE GEOPOLITICAL SPIKE", fontsize=14)
    plt.ylabel("Lead Time (Days)")
    plt.axhline(y=3, color='r', linestyle='--', label='Critical Warning Threshold')
    plt.legend()
    plt.show()