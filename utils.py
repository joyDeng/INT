import numpy as np

sum = 0.0
length = 0
for i in range(3):
    gradients = np.load(f"gradient_fd_{i}.npy")
    print(gradients)
    sum += np.sum(gradients)
    length += gradients.shape[0]
 
print("avg", sum / length)
