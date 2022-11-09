import numpy as np
import matplotlib.pyplot as plt

x = ['-400', '-300', '-200', '-100', '0', '100', '200', '300', '400']
y = ['-75', '25', '125', '225', '325', '425']
# y = y[::-1]

errors = [[12.21, 9.17, 9.43, 6.71, 8.83, 7.81], 
          [11.36, 7.81, 5.20, 3.16, 4.36, 7.00],
          [9.80, 5.83, 2.45, 2.24, 3.32, 4.12],
          [9.27, 5.10, 3.74, 0.00, 1.73, 5.10], 
          [0.0, 0.0, 3.32, 3.16, 2.00, 1.00],
          [13.64, 8.25, 6.78, 5.00, 3.74, 2.24],
          [15.78, 13.15, 9.64, 7.62, 5.74, 5.74],
          [19.52, 15.81, 13.60, 10.82, 9.64, 7.35],
          [21.21, 18.49, 16.67, 14.21, 10.05, 8.06]]
errors = np.array(errors).T.astype(int)
# print(errors)
# errors = errors[::-1]
# print(errors)

fig, ax = plt.subplots()
plt.pcolor(x, y, errors, cmap='RdBu', vmin=errors.max(), vmax=errors.min())
plt.colorbar()
# im = ax.imshow(errors)
# ax.set_xticks(np.arange(len(x)), labels=x)
# ax.set_yticks(np.arange(len(y)), labels=y)

for i in range(len(y)):
    for j in range(len(x)):
        text = plt.text(j, i, errors[i][j], ha="center", va="center", color="w")
ax.set_title("Calibration Error heatmap")
fig.tight_layout()
plt.show()