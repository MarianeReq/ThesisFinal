import numpy as np

# Convert actual and predicted values to arrays
actual_values = np.array([ 
                          6.05, 
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 5.9545, 5.9545,
                          5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 
                          5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545,
                          5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545])

predicted_results = np.array([8.549149513, 8.508260727, 8.561875343, 8.479343414, 8.48433876, 
                               8.445913315, 8.515300751, 8.48938179, 8.526301384, 8.501797676, 
                               8.51830864, 8.540571213, 8.486351967, 8.448896408, 8.512660027, 
                               6.047782421, 8.535227776, 8.464323997, 8.500367165, 8.52508831, 
                               8.452683449, 8.519338608, 8.46185112, 8.446587563, 8.458095551, 
                               8.48515892, 8.520030022, 8.445696831, 8.517214775, 8.423130989, 
                               8.413051605, 8.431301117, 8.420885086, 8.47875309, 8.454007149, 
                               8.44540596, 8.531791687, 8.480399132, 8.491340637, 8.458618164, 
                               8.516368866, 8.50940609, 8.47118187, 8.377340317, 8.431030273, 
                               8.436617851, 8.663976192, 7.754868031, 6.170273304, 6.169104099, 
                               6.157329082, 6.162255287, 6.173491955, 6.166065693, 6.736694813, 
                               6.691514015, 6.676287174, 6.64495039, 6.658804417, 6.635145187, 
                               6.650177956, 6.663976192, 6.665077209, 6.649903774, 6.887138844, 
                               6.87058115, 6.868009567, 8.051041603, 7.913097858, 7.929887295, 
                               7.938713074, 7.905397892, 7.911950588, 7.894933701, 7.892903328, 
                               7.888812065, 7.923159599, 7.922402382, 7.927269936, 7.883018494, 
                               7.926161766, 7.894525528, 7.915867805, 7.955296993, 7.729655743])

# Threshold for classification
threshold = 6.05

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

