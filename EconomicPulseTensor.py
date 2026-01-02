def get_strategic_weights(current_date):
    # Days remaining until the Aug 1, 2027 Centennial
    deadline = pd.to_datetime("2027-08-01")
    days_left = (deadline - current_date).days

    # Sigmoid function to increase maneuver 'urgency' as 2027 approaches
    readiness_weight = 1 / (1 + np.exp(days_left / 365))

    return readiness_weight

# Example: On Jan 1, 2026, the readiness_weight is significantly
# higher than it was in 2024, priming the LSTM for 'aggressive' behavior.