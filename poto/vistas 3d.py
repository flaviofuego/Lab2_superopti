import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('dataset\Student_Performance.csv')

# Function to create an interactive 3D plot
def interactive_3d_plot(data, x1_name, x2_name, y_name, title):
    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the data
    x1 = data[x1_name]
    x2 = data[x2_name]
    y = data[y_name]
    
    # Create the scatter plot
    scatter = ax.scatter(x1, x2, y, c=y, cmap=cm.viridis, 
                        s=50, alpha=0.6, edgecolors='w', linewidth=0.5)
    
    # Fit a regression model to get the plane
    X = data[[x1_name, x2_name]]
    model = LinearRegression()
    model.fit(X, y)
    
    # Create a mesh for the regression plane
    x1_range = np.linspace(x1.min(), x1.max(), 20)
    x2_range = np.linspace(x2.min(), x2.max(), 20)
    x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
    
    # Generate predictions for the mesh
    X_mesh = np.column_stack((x1_mesh.flatten(), x2_mesh.flatten()))
    y_pred = model.predict(X_mesh).reshape(x1_mesh.shape)
    
    # Plot the regression plane
    surf = ax.plot_surface(x1_mesh, x2_mesh, y_pred, alpha=0.3, 
                          cmap=cm.coolwarm, linewidth=0, antialiased=True)
    
    # Add labels and title
    ax.set_xlabel(f'{x1_name}', fontsize=12, labelpad=10)
    ax.set_ylabel(f'{x2_name}', fontsize=12, labelpad=10)
    ax.set_zlabel(f'{y_name}', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16, pad=20)
    
    # Show the equation of the plane
    intercept = model.intercept_
    coef1 = model.coef_[0]
    coef2 = model.coef_[1]
    equation = f'{y_name} = {coef1:.2f}*{x1_name} + {coef2:.2f}*{x2_name} + {intercept:.2f}'
    r2 = model.score(X, y)
    
    fig.text(0.15, 0.85, f'Equation: {equation}\nR² = {r2:.4f}', 
            fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add a color bar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label(f'{y_name}', rotation=270, labelpad=20, fontsize=12)
    
    return fig, ax

# Create an interactive plot
fig, ax = interactive_3d_plot(
    data, 
    'Previous Scores', 
    'Hours Studied', 
    'Performance Index',
    'Interactive 3D: Previous Scores and Hours Studied vs Performance'
)

# This is important - it makes the window interactive and non-blocking
plt.show(block=False)

# You can create another one
fig2, ax2 = interactive_3d_plot(
    data, 
    'Hours Studied', 
    'Sleep Hours', 
    'Performance Index',
    'Interactive 3D: Hours Studied and Sleep Hours vs Performance'
)

# Show all windows and keep the script running
plt.show()

# Note: You can rotate, zoom, and explore all visualizations using your mouse