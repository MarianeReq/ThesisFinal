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
                          0.61, 0.61, 0.61, 0.61, 0.61])

predicted_results = np.array([0.8957926035, 0.9055423737, 0.7823274136, 0.4736533165, 0.7122320533,
                               0.9792575836, 0.799228549, 0.4269301295, 0.7414009571, 0.7127989531,
                               0.7504867315, 0.742010355, 0.7417484522, 0.6206240654, 0.8971276283,
                               0.9000689983, 0.8946923018, 0.8938097954, 0.8951020241, 0.6220605373,
                               0.6218146682, 0.8950141668, 0.8957587481, 0.9072012901, 0.9066262245,
                               0.6224523783, 0.7401871681, 0.6199797988, 0.9105424881, 0.9045151472,
                               0.9080872536, 0.9104471207, 0.7851503491, 0.7394660711, 0.7820230126,
                               0.9114772081, 0.9076448679, 0.9146502018, 0.9048435688, 0.7808276415,
                               0.7414472103, 0.7788822651, 0.9118359089, 0.9122989178, 0.9040048122,
                               0.9073046446, 0.7812678814, 0.7414981127, 0.7421731949, 0.7801843882,
                               0.9136148691, 0.9055029154, 0.9120444059, 0.7847572565, 0.7384973764,
                               0.7790161371, 0.9080078602, 0.9104565382, 0.9055392742, 0.780809164,
                               0.7391511202, 0.7377998233, 0.779307723, 0.7764555216, 0.7802107334,
                               0.7800642848, 0.7793890238, 0.7384623885, 0.7398388386, 0.7779502869,
                               0.7754206657, 0.6681150794, 0.6719033718, 0.6711159348, 0.738147378,
                               0.742028296, 0.6682889462, 0.6723302603, 0.6687760353, 0.6719388962,
                               0.6742472649, 0.7424441576, 0.7440098524, 0.6712511778, 0.7874041796])

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
