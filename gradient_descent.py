import matplotlib.pyplot as plt
import numpy as np

dataset = {
    3.5: 18,
    3.69: 15,
    3.44: 18,
    3.43: 16,
    4.34: 15,
    4.42: 14,
    2.37: 24,
}

# Initialize weight, bias and learning rate
weight = 0
bias = 0
learning_rate = 0.06958255

# Initialize convergence criteria, previous MSE, and maximum iterations
convergence_threshold = 0.0000001  # Stop when MSE change is less than this
previous_mse = float('inf')
max_iterations = 50000  # prevents infinite loops

for iteration in range(max_iterations):
    total_error = 0
    weight_gradient = 0
    bias_gradient = 0
    
    # Calculate gradients and error using current parameters
    for x, y_actual in dataset.items():
        error = bias + weight * x - y_actual
        total_error += error ** 2
        
        weight_gradient += 2 * error * x
        bias_gradient += 2 * error
    
    # Calculate current MSE
    current_mse = total_error / len(dataset)
    
    # Print current iteration results
    print(f"Iteration {iteration+1}: weight={weight:.2f}, bias={bias:.2f}, MSE={current_mse:.2f}")
    
    # Check for convergence
    if abs(previous_mse - current_mse) < convergence_threshold:
        print(f"\nConverged after {iteration+1} iterations!")
        break
    
    previous_mse = current_mse
    
    # Update parameters for next epoch
    weight -= learning_rate * weight_gradient / len(dataset)
    bias -= learning_rate * bias_gradient / len(dataset)
    
    # Safety check for NaN values
    if any(val != val for val in [weight, bias, current_mse]):
        print("NaN values detected - stopping training")
        break

else:
    print(f"\nReached maximum iterations ({max_iterations}) without full convergence")

# Plotting with regression line
x_values = list(dataset.keys())
y_values = list(dataset.values())

plt.figure(figsize=(12, 8))

# Plot data points
plt.scatter(x_values, y_values, color='blue', s=100, label='Data points', zorder=5)

# Plot regression line
x_min, x_max = min(x_values), max(x_values)
x_line = np.linspace(x_min - 0.5, x_max + 0.5, 100)
y_line = bias + weight * x_line
plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Regression line: y = {weight:.2f}x + {bias:.2f}')

# Plot predictions vs actuals
for x, y_actual in dataset.items():
    y_pred = bias + weight * x
    plt.plot([x, x], [y_actual, y_pred], color='gray', linestyle='--', alpha=0.7)

plt.xlabel('X values')
plt.ylabel('Y values')
plt.title(f'Linear Regression Results (MSE: {current_mse:.2f})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()