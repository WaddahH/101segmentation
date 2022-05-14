import numpy as np
from sklearn.cluster import KMeans
import cv2
from matplotlib import pyplot as plt

image1 = cv2.imread("mri1.jpg")
image2 = cv2.imread("mri2.jpg")
image3 = cv2.imread("Berlin.jpg")
image = [image1, image2, image3]
reshaped = [0, 0, 0]
for imgge in range(0, 3):
    reshaped[imgge] = image[imgge].reshape(image[imgge].shape[0] * image[imgge].shape[1], image[imgge].shape[2])

numClusters = list(map(int, input("Enter the number of culsters for all imges respectively: ").split(" ")))

clustering = [0, 0, 0]
for imgge in range(0, 3):
    kmeans = KMeans(n_clusters=numClusters[imgge], n_init=40, max_iter=500).fit(reshaped[imgge])
    clustering[imgge] = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
                                   (image[imgge].shape[0], image[imgge].shape[1]))

sortedLabels = [[], [], []]
for imgge in range(0, 3):
    sortedLabels[imgge] = sorted([n for n in range(numClusters[imgge])],
                                 key=lambda x: -np.sum(clustering[imgge] == x))

kmeansImage = [0, 0, 0]
concatImage = [[], [], []]
for kmeeens in range(0, 3):
    kmeansImage[kmeeens] = np.zeros(image[kmeeens].shape[:2], dtype=np.uint8)
    for imgge, label in enumerate(sortedLabels[kmeeens]):
        kmeansImage[kmeeens][clustering[kmeeens] == label] = int((255) / (numClusters[kmeeens] - 1)) * imgge
    concatImage[kmeeens] = np.concatenate((image[kmeeens], 193 * np.ones((image[kmeeens].shape[0], int(0.0625 * image[kmeeens].shape[1]), 3),
                                                                         dtype=np.uint8),
                                           cv2.cvtColor(kmeansImage[kmeeens], cv2.COLOR_GRAY2BGR)), axis=1)

plt.imshow(concatImage[0])
plt.show()
plt.imshow(concatImage[1])
plt.show()
plt.imshow(concatImage[2])
plt.show()
