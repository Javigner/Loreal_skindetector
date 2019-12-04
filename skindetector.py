import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
from matplotlib import pyplot as plt
from math import sqrt
import pandas as pd


def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=5, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar

def find_nearest(df, skin_color):
    result = df.iloc[0]
    for index, row in df.iterrows():
        tmp = sqrt(pow(skin_color[0] - row['r'], 2) + pow(skin_color[1] - row['g'], 2) + pow(skin_color[2] - row['b'], 2))
        if (index == 0):
            euclidian = sqrt(pow(skin_color[0] - row['r'], 2) + pow(skin_color[1] - row['g'], 2) + pow(skin_color[2] - row['b'], 2))
        if (tmp < euclidian):
            result = row
            euclidian = tmp
    return result

def find_nearest_v2(df, skin_color):
    result = df[0][2]
    euclidian = sqrt(pow(skin_color[0] - float(df[0][2]), 2) + pow(skin_color[1] - float(df[0][3]), 2) + pow(skin_color[2] - float(df[0][4]), 2))
    for element in df:
        tmp = sqrt(pow(skin_color[0] - float(element[2]), 2) + pow(skin_color[1] - float(element[3]), 2) + pow(skin_color[2] - float(element[4]), 2))
        if (tmp < euclidian):
            result = element
            euclidian = tmp
    return result

if __name__ == "__main__":
    image = cv2.imread("Elise.jpg")
    image = cv2.imencode('.jpg', image)[1].tostring()
    nparr = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = imutils.resize(image, width=250)
    plt.subplot(3, 1, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    skin = extractSkin(image)
    plt.subplot(3, 1, 2)
    plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
    plt.title("Thresholded  Image")
    dominantColors = extractDominantColor(skin, hasThresholding=True)
    colour_bar = plotColorBar(dominantColors)
    result = dominantColors[0]['color']
    color_list = [['Alabaster', '#DEBB9F', '222', '187', '159'], ['Beige', '#C59567', '197', '149', '103'], ['Cappuccino', '#935A37', '147', '90', '55'], ['Caramel', '#B68962', '182', '137', '98'], ['Deep Cool', '#67412C', '103', '65', '44'], ['Deep Espresso', '#563025', '86', '48', '37'], ['Deep Walnut', '#623522', '98', '53', '34'], ['Fair', '#DFC3AB', '223', '195', '171'], ['Golden', '#BB8758', '187', '135', '88'], ['Golden Honey', '#BF8357', '191', '131', '87'], ['Light Ivory', '#EFC6AA', '239', '198', '170'], ['Mahogany', '#A06943', '160', '105', '67'], ['Medium Olive', '#CE9971', '206', '153', '113'], ['Mocha', '#78452A', '120', '69', '42'], ['Natural', '#DBA380', '219', '163', '128'], ['Neutral Buff', '#C4885E', '196', '136', '94'], ['Neutral Tan', '#AF794D', '175', '121', '77'], ['Pale', '#F6D3AD', '246', '211', '173'], ['Soft Beige', '#CE9367', '206', '147', '103'], ['True Beige', '#DBA776', '219', '167', '118'], ['Vanilla', '#E9BB99', '233', '187', '153'], ['Walnut', '#643A21', '100', '58', '33'], ['Warm Caramel', '#A0603A', '160', '96', '58'], ['Warm Honey', '#AB7046', '171', '112', '70']]
    nearest_color = find_nearest_v2(color_list,dominantColors[0]['color'])
    nearest_color2 = find_nearest_v2(color_list,dominantColors[1]['color'])
    print(nearest_color)
    #color = pd.read_csv('color.csv')
    #nearest_color = find_nearest(color, dominantColors[0]['color'])
    #nearest_color2 = find_nearest(color, dominantColors[1]['color'])
    plt.subplot(3, 1, 3)
    plt.axis("off")
    plt.imshow(colour_bar)
    plt.title("Color Bar")
    plt.tight_layout()
    plt.show()
    df_color = [nearest_color[0], nearest_color2[0]]
    print(df_color)
