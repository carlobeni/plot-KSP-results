import matplotlib.pyplot as plt

# List of KSP methods to plot
ksp_methods = ["gmres", "lgmres"]
residual_norms = {}

# Read residual norms from files
for method in ksp_methods:
    with open(f"residual_norms_python/value_{method}.txt", 'r') as file:
        iterations = []
        norms = []
        for line in file:
            iteration, norm = line.split()
            iterations.append(int(iteration))
            norms.append(float(norm))
        residual_norms[method] = (iterations, norms)

# Plotting
#plt.figure(figsize=(10, 6))

for method in ksp_methods:
    iterations, norms = residual_norms[method]
    plt.plot(iterations, norms, label=method)

plt.yscale('log')  # Set log scale for better visualization of norms
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.title('Convergence of Different KSP Methods')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('ksp_methods_convergence.png')
plt.show()
