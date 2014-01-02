import numpy as np
import cv2

def resize_60x60(
    px_array,              # image array
    bound_threshold=0.02,  # density threshold to make new bounding box
    bw_threshold=0.04,     # to black/white density threhold
):
    """resize image to 60x60 pixels and black/white-ize.

    Parameters
    ==========
    bound_threshold : 0.02, float
        density threshold for new bounding box(large white span removed)

    bw_threshold : 0.04, float
        density threshold to black/white-ize, so only 0/1 will return
        by setting it to -1, this step will be skipped
    """

    img = px_array.reshape((122, 105))

    # bounding box
    blackpoints = np.transpose((img >= bound_threshold).nonzero())
    rect = cv2.boundingRect(np.array([blackpoints], dtype=np.int32))
    img_bounding = img[
        max(rect[0] - 1, 0):min(rect[0] + rect[2] + 2, 121),
        max(rect[1] - 1, 0):min(rect[1] + rect[3] + 2, 104)
    ]

    # resize
    img_6060 = cv2.resize(img_bounding, (60, 60))

    # threshold
    if bw_threshold > 0:
        img_threshold = (img_6060 >= bw_threshold).astype('int')
    else:
        img_threshold = img_6060

    return img_threshold
