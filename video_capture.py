import cv2
import numpy as np
from sklearn.cluster import KMeans
def process_frame(frame, k=4):
    frame = cv2.resize(frame, (320, 240))  # Resize for speed
    original = frame.copy()

    # Flatten and cluster
    pixels = frame.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
    clustered = kmeans.cluster_centers_[kmeans.labels_].reshape(frame.shape).astype(np.uint8)

    # Convert to grayscale for contour detection
    gray = cv2.cvtColor(clustered, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect contours and draw boxes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 30 and h > 30:
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return original
cap = cv2.VideoCapture("C:/Users/koush/Videos/Screen Recordings/Screen Recording 2025-05-26 165110.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = process_frame(frame, k=4)
    cv2.imshow("AR Simulation from Recorded Video", result)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()