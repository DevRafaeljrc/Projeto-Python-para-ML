import matplotlib

from sklearn.datasets import make_regression
x, y = make_regression(n_samples=200, n_features=1, noise=30)
import matplotlib.pyplot as plt
plt.scatter(x, y, color='blue')
plt.title('Scatter plot of x and y')
plt.xlabel('x')
plt.ylabel('y') 
plt.show()

