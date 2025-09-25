import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('out/tables/exp_10_ga_stats.csv')

# Plot fitness evolution
plt.figure(figsize=(10, 6))
plt.plot(data['gen'], data['avg'], label='Average Fitness', color='blue')
plt.plot(data['gen'], data['min'], label='Minimum Fitness', color='red', linestyle='--')
plt.plot(data['gen'], data['max'], label='Maximum Fitness', color='green', linestyle='--')

# Add labels, title, and legend
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Evolution Over Generations')
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig('best_fitness_evolution.png')
plt.show()

# Plot number of evaluations over generations
plt.figure()
plt.plot(data['gen'], data['nevals'], label='Number of Evaluations', color='purple')

# Add labels, title, and legend
plt.xlabel('Generation')
plt.ylabel('Number of Evaluations')
plt.title('Number of Evaluations Over Generations')
plt.legend()
plt.grid(True)

# Save and show the plot
plt.savefig('number_of_evaluations.png')
plt.show()