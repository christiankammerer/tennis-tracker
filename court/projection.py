import cv2
import numpy as np
import matplotlib.pyplot as plt

corners = [(279.6, 502.8), (1002.4, 511.8), (824.7, 167.8), (458.3, 164.8)]
def project_image(image, corners):
    image_height, image_width = image.shape[:2]
    corners = np.array(corners)
    corners = corners.reshape(-1, 1, 2)
    corners = corners.astype(np.float32)
    corners = corners.reshape(1, 4, 2)
    corners = corners.astype(np.float32)
    corners = corners.reshape(4, 2)
    corners = corners.astype(np.float32)
    corners = corners.reshape(1, 4, 2)
    corners = corners.astype(np.float32)
    corners = corners.reshape(4, 2)
    print(corners)

def __main__():
    image = cv2.imread("debug/frame_0.jpg")
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.axis("off")
    plt.show()
    
