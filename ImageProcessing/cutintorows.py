import cv2


def detect_rows_full(image):
    # cv2.imshow(" ", image)
    # cv2.waitKey(0)
    # 扩大黑色面积，使效果更明显
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # handled = cv2.erode(image, kernel1, iterations=1)  # 先腐蚀
    handled = image
    height, width = handled.shape[:2]
    z = [0] * height
    # 水平投影并统计每一行的黑点数
    a = 0
    for y in range(0, height):
        for x in range(0, width):
            if handled[y, x] < 100:
                a = a + 1
            else:
                continue
        z[y] = a
        a = 0
    begin = False
    lastH = 0
    h_list = []
    division = []
    for y in range(0, height):
        if z[y] > 0:
            begin = True
        elif begin:
            h_list.append(y - lastH)
            division.append((lastH, y))
            lastH = y
            begin = False
        else:
            lastH = y
    if z[height - 1] > 0:
        h_list.append(height - lastH)
        division.append((lastH, height))
    return len(h_list) > 1, division
