import tensorflow as tf
import numpy as np

# Example data (replace this with your actual data)
features = np.array([[[1.0, 2.0], [3.0, 4.0]], [[3.0, 4.0], [5.0, 6.0]], [[5.0, 6.0], [1.0, 2.0]]]) # 3 * 2 * 2
labels = np.array([[[0], [1]], [[1], [0]], [[1], [1]]]) # 3 * 2 * 1

# Create a dataset from tensors using from_tensor_slices
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

print(dataset)

y = np.concatenate([y for x, y in dataset], axis=0)

print(y)

# Print the elements in the dataset
for element in dataset:
    print("Features:", element[0].numpy(), "Label:", element[1].numpy())
