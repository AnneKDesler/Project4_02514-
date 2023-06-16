import selectivesearch
import cv2

def SS(img):
    """
    Performs Selective Search on an image
    :param img: image to be processed
    :return: img_lbl: image with labels
             regions: regions of the image
    """
    img_lbl, regions = selectivesearch.selective_search(img)#, scale=500, sigma=0.9, min_size=10)

    return img_lbl, regions


if __name__ == "__main__":
    path = "/u/data/s194333/DLCV/Project4_02514-/data/batch_1/000000.jpg"
    img = cv2.imread(path)
    # resize image to 512x512
    img = cv2.resize(img, (512, 512))
    print(img.shape)
    print(img.dtype)
    print(img.max())
    img_lbl, regions = SS(img)
    print(regions)
    print(len(regions))
    print(regions[0])

    # draw rectangles on the original image
    for r in regions:
        x, y, w, h = r['rect']
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imwrite("img.png", img)


