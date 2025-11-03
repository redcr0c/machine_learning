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

def train_with_learning_rate(learning_rate, dataset, max_iterations=50000):
    """
    Train the model with given learning rate and return number of iterations to converge
    """
    weight = 0
    bias = 0
    convergence_threshold = 0.0000001
    previous_mse = float('inf')
    
    for iteration in range(max_iterations):
        total_error = 0
        weight_gradient = 0
        bias_gradient = 0
        
        for x, y_actual in dataset.items():
            error = bias + weight * x - y_actual
            total_error += error ** 2
            weight_gradient += 2 * error * x
            bias_gradient += 2 * error
        
        current_mse = total_error / len(dataset)
        
        if abs(previous_mse - current_mse) < convergence_threshold:
            return iteration + 1, weight, bias, current_mse, True
        
        previous_mse = current_mse
        
        weight -= learning_rate * weight_gradient / len(dataset)
        bias -= learning_rate * bias_gradient / len(dataset)
        
        if any(val != val for val in [weight, bias, current_mse]) or current_mse > 1e10:
            return max_iterations, weight, bias, current_mse, False
    
    return max_iterations, weight, bias, current_mse, False

def find_optimal_learning_rate_precise():
    """
    Find the optimal learning rate using a multi-stage precision approach
    """
    
    all_results = []
    
    lr_range1 = np.logspace(-3, 0.3, 50)
    stage1_results = []
    
    for lr in lr_range1:
        iterations, weight, bias, mse, converged = train_with_learning_rate(lr, dataset)
        result = {
            'learning_rate': lr,
            'iterations': iterations,
            'weight': weight,
            'bias': bias,
            'mse': mse,
            'converged': converged
        }
        stage1_results.append(result)
        all_results.append(result)
            
    converged_stage1 = [r for r in stage1_results if r['converged']]
    if not converged_stage1:
        lr_range1 = np.logspace(-4, 1, 100)
        stage1_results = []
        for lr in lr_range1:
            iterations, weight, bias, mse, converged = train_with_learning_rate(lr, dataset)
            result = {'learning_rate': lr, 'iterations': iterations, 'converged': converged}
            stage1_results.append(result)
            all_results.append(result)
        converged_stage1 = [r for r in stage1_results if r['converged']]
    
    if not converged_stage1:
        return None, all_results
    
    best_stage1 = min(converged_stage1, key=lambda x: x['iterations'])
  
    center_lr = best_stage1['learning_rate']
    
    lr_range2 = np.linspace(max(0.001, center_lr * 0.5), center_lr * 1.5, 30)
    stage2_results = []
    
    for lr in lr_range2:
        iterations, weight, bias, mse, converged = train_with_learning_rate(lr, dataset)
        result = {
            'learning_rate': lr,
            'iterations': iterations,
            'weight': weight,
            'bias': bias,
            'mse': mse,
            'converged': converged
        }
        stage2_results.append(result)
        all_results.append(result)
    
    converged_stage2 = [r for r in stage2_results if r['converged']]
    best_stage2 = min(converged_stage2, key=lambda x: x['iterations']) if converged_stage2 else best_stage1
    
    center_lr = best_stage2['learning_rate']
    lr_range3 = np.linspace(center_lr * 0.9, center_lr * 1.1, 50)
    stage3_results = []
    
    for lr in lr_range3:
        iterations, weight, bias, mse, converged = train_with_learning_rate(lr, dataset)
        result = {
            'learning_rate': lr,
            'iterations': iterations,
            'weight': weight,
            'bias': bias,
            'mse': mse,
            'converged': converged
        }
        stage3_results.append(result)
        all_results.append(result)
 
    converged_stage3 = [r for r in stage3_results if r['converged']]
    best_stage3 = min(converged_stage3, key=lambda x: x['iterations']) if converged_stage3 else best_stage2
        
    center_lr = best_stage3['learning_rate']
    lr_range4 = np.linspace(center_lr * 0.99, center_lr * 1.01, 30)
    stage4_results = []
    
    for lr in lr_range4:
        iterations, weight, bias, mse, converged = train_with_learning_rate(lr, dataset)
        result = {
            'learning_rate': lr,
            'iterations': iterations,
            'weight': weight,
            'bias': bias,
            'mse': mse,
            'converged': converged
        }
        stage4_results.append(result)
        all_results.append(result)
    
    converged_stage4 = [r for r in stage4_results if r['converged']]
    optimal_result = min(converged_stage4, key=lambda x: x['iterations']) if converged_stage4 else best_stage3
    
    return optimal_result, all_results

def plot_precise_analysis(results, optimal_result):
    """
    Plot the precise learning rate analysis
    """
    converged_results = [r for r in results if r['converged']]
    
    if not converged_results:
        return
    
    lrs = [r['learning_rate'] for r in converged_results]
    iterations = [r['iterations'] for r in converged_results]
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(lrs, iterations, alpha=0.6, s=50, color='blue', label='All converged LRs')
    
    plt.scatter(optimal_result['learning_rate'], optimal_result['iterations'], 
                color='red', s=200, marker='*', edgecolors='black', 
                label=f'Optimal: LR={optimal_result["learning_rate"]:.8f}')
    
    plt.xlabel('Learning Rate')
    plt.ylabel('Iterations to Converge')
    plt.title(f'Precise Learning Rate Optimization\nOptimal: {optimal_result["learning_rate"]:.8f} → {optimal_result["iterations"]} iterations')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    min_iters = min(iterations)
    max_iters = max(iterations)
    avg_iters = np.mean(iterations)
    
    plt.figtext(0.02, 0.02, 
                f'Statistics:\nMin iterations: {min_iters}\nMax iterations: {max_iters}\nAverage iterations: {avg_iters:.1f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    optimal_result, all_results = find_optimal_learning_rate_precise()
    
    if optimal_result:
        print(f"Optimal LR: {optimal_result['learning_rate']:.10f}")
        print(f"Iterations: {optimal_result['iterations']}")
        print(f"Final Weight: {optimal_result['weight']:.6f}")
        print(f"Final Bias: {optimal_result['bias']:.6f}")
        print(f"Final MSE: {optimal_result['mse']:.6f}")
        
        # plot_precise_analysis(all_results, optimal_result)

    else:
        print("Failed to find an optimal learning rate!")