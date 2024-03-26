import numpy as np

# Convert actual and predicted values to arrays
actual_values = np.array([0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 
                          0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 
                          0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 
                          0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 
                          0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62, 
                          0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.67, 0.67, 0.66, 0.66, 
                          0.66, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 
                          0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 
                          0.61, 0.61, 0.61, 0.61, 0.61 ])

predicted_results = np.array([0.780184269, 0.7792118192, 0.7869827747, 0.778036356, 0.7650943995, 
                               0.6674919128, 0.6710397005, 0.6697906852, 0.6653821468, 0.6632524729, 
                               0.6675161719, 0.6690630913, 0.6542596221, 0.6712992191, 0.6669201255, 
                               0.4703083634, 0.6646437645, 0.6670130491, 0.6670291424, 0.6673668623, 
                               0.6694891453, 0.6671311855, 0.6689229012, 0.6696721315, 0.6676920652, 
                               0.6673061848, 0.6630828977, 0.6653453112, 0.6691776514, 0.6691782475, 
                               0.6719833016, 0.6696045399, 0.6674740314, 0.6668967009, 0.664129436, 
                               0.6695349216, 0.6659493446, 0.6649422646, 0.670574069, 0.6670895815, 
                               0.6722505093, 0.6658092737, 0.66488868, 0.6677212715, 0.6666896343, 
                               0.6674021482, 0.5765314102, 0.5785529613, 0.669298768, 0.6676014662, 
                               0.6689633131, 0.6683856249, 0.665417254, 0.6711417437, 0.7020803094, 
                               0.6979228258, 0.6996760368, 0.6972357035, 0.697599113, 0.6980386972, 
                               0.6975054741, 0.6962375641, 0.6984227896, 0.6962993145, 0.7006494999, 
                               0.6978131533, 0.6996931434, 0.7006336451, 0.702008605, 0.6962836981, 
                               0.6979893446, 0.6996836662, 0.6983456612, 0.7020441294, 0.7018917799, 
                               0.7026646137, 0.7011605501, 0.7012096643, 0.6994231343, 0.6992453337, 
                               0.6991759539, 0.7027183771, 0.6912433505, 0.6962388158, 0.6948894262])

# Threshold for classification
threshold = 0.5

# Convert probabilities to binary predictions
binary_predictions = np.where(predicted_results >= threshold, 1, 0)

# Calculate True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
TP = np.sum((actual_values >= threshold) & (binary_predictions == 1))
TN = np.sum((actual_values < threshold) & (binary_predictions == 0))
FP = np.sum((actual_values < threshold) & (binary_predictions == 1))
FN = np.sum((actual_values >= threshold) & (binary_predictions == 0))

# Calculate accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Calculate precision
precision = TP / (TP + FP)

# Calculate recall
recall = TP / (TP + FN)

# Calculate F1-score
f1_score = 2 * (precision * recall) / (precision + recall)

print("True Positives (TP):", TP)
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
