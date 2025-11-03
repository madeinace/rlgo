import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Box

# Define Box space
space = Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)

# Sample some points
samples = np.array([space.sample() for _ in range(200)])

# Plot bounding rectangle
plt.figure(figsize=(6, 6))
rect = plt.Rectangle(
    xy=space.low,  # bottom-left corner (-1, -2)
    width=space.high[0] - space.low[0],  # width from x min to x max
    height=space.high[1] - space.low[1],  # height from y min to y max
    fill=False,
    color="red",
    linewidth=2,
    label="Box Region",
)
plt.gca().add_patch(rect)

# Plot samples
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, label="Random Samples")

# Labels & limits
plt.xlim(space.low[0] - 0.5, space.high[0] + 0.5)
plt.ylim(space.low[1] - 0.5, space.high[1] + 0.5)
plt.xlabel("Dimension 0")
plt.ylabel("Dimension 1")
plt.title("Visualization of 2D Box Space")
plt.legend()
plt.grid(True)
plt.show()
