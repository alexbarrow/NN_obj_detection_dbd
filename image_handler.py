import xml.etree.cElementTree as ET
import cv2
import matplotlib.pyplot as plt

BOX_KILLER_COLOR = (255, 0, 0)  # Red
BOX_SURV_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)  # White

id_to_labels = {1: 'killer', 2: 'surv'}


def get_bb_list(root_path):
    tree = ET.ElementTree(file=root_path)

    root = tree.getroot()
    bb_list = []
    class_labels = []

    for child_of_root in root.iter('object'):

        bb_temp = []
        for bb in child_of_root.iter('bndbox'):
            child = bb.getchildren()
            for ch in child:
                bb_temp.append(int(ch.text))

        for name in child_of_root.iter('name'):
            label = name.text
            class_labels.append(label)
        bb_list.append(bb_temp)

    return bb_list, class_labels


def visualize_bbox(img, bbox, class_name, color, thickness=1):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, labels):
    img = image.copy()
    for bbox, label in zip(bboxes, labels):
        class_name = id_to_labels[label]

        if class_name == 'surv':
            color = BOX_SURV_COLOR
        elif class_name == 'killer':
            color = BOX_KILLER_COLOR
        else:
            color = TEXT_COLOR

        img = visualize_bbox(img, bbox, class_name, color)

    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img.astype('uint8'))
    plt.show()


if __name__ == '__main__':
    root_p = 'data/test_file.xml'
    print(get_bb_list(root_p))
