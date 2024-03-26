import numpy as np

# Convert actual and predicted values to arrays
actual_values = np.array([
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 
                          6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 6.05, 
                          6.05, 6.05, 6.05, 6.05, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 
                          5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 
                          5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 5.9545, 
                          5.9545, 5.9545, 5.9545, 5.9545])

predicted_results = np.array([8.024265289, 9.095118523, 10.87127495, 10.84157562, 14.04133606,
                              9.297359467, 11.15234184, 6.051625252, 10.85571194, 6.873292446,
                              9.377877235, 10.80870819, 9.358011246, 9.355002403, 9.376496315,
                              9.373976707, 9.42354393, 10.8336134, 8.008791924, 9.393685341,
                              9.35278511, 9.432584763, 10.85345554, 10.85623741, 10.81846142,
                              10.8686018, 10.8871603, 10.81937218, 10.84386253, 9.404259682,
                              9.0664711, 9.074052811, 10.90709114, 10.29127693, 10.25662136,
                              10.23733711, 10.43453026, 10.24386978, 10.84715843, 8.55103302,
                              8.564992905, 8.584432602, 8.575131416, 8.586036682, 10.8318243,
                              10.83790016, 6.099005222, 6.034731865, 6.018235207, 6.044029236,
                              6.06965971, 6.058913231, 9.085935593, 9.380311966, 10.57567215,
                              10.58445358, 10.52200794, 10.54988098, 6.022107124, 10.84259605,
                              7.73814249, 7.792140484, 7.751787186, 6.040099621, 10.41588783,
                              10.82519722, 10.90345955, 10.88169575, 10.89161873, 10.88124275,
                              10.84928417, 10.89166355, 10.88153648, 6.065125942, 10.81064224,
                              10.84118938, 10.91062927, 10.85589027, 10.85672283, 10.86089516,
                              6.048820019, 10.86268425, 10.85145664, 10.8811636, 10.84355927])

# Threshold for classification
threshold = 6.0

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
