import cv2
import numpy as np

def label_segments_by_distance(image, contours, segment_length=100):
    segment_id = 1
    segment_positions = []

    for contour in contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        total_distance = 0
        for i in range(1, len(approx)):
            p1 = approx[i - 1][0]
            p2 = approx[i][0]

            distance = np.linalg.norm(p1 - p2)
            total_distance += distance

            if total_distance >= segment_length:
                mid_point = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
                segment_positions.append((mid_point, segment_id))
                segment_id += 1
                total_distance = 0

                # Draw the segment
                cv2.line(image, tuple(p1), tuple(p2), (0, 255, 0), 2)

    for pos, segment_id in segment_positions:
        cv2.putText(image, f'Segment {segment_id}', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Load the image
image_path = 'duzasagiikiyol.jpeg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Label segments based on distance
label_segments_by_distance(image, contours, segment_length=150)

# Show the result
cv2.imshow('Detected Segments', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
