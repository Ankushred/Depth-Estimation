import cv2
import matplotlib.pyplot as plt

depth_map = cv2.imread("Grayscale test1.jpg", cv2.IMREAD_UNCHANGED)
image = cv2.imread("test1.jpg", cv2.IMREAD_UNCHANGED)
plt.imshow(image)
plt.title("Click on Two Points")
plt.axis("off")
points = plt.ginput(n=2, timeout=30,show_clicks=True)  
plt.close()  

if len(points) == 2:
    p1 = (int(points[0][0]), int(points[0][1]))  # (x1, y1)
    p2 = (int(points[1][0]), int(points[1][1]))  # (x2, y2)

    D1 = depth_map[p1[1], p1[0]]
    D2 = depth_map[p2[1], p2[0]]

    if D1 > D2:
        closer, farther = "Point 1","Point 2"
    else:
        closer, farther = "Point 2","Point 1"

    print(f"🔹 The point {closer} is closer.")
    print(f"🔹 The point {farther} is farther.")

else:
    print("Error: You must select exactly two points!")
