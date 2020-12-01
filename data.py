import os
import cv2
import torch
import numpy as np

from torch.utils.data import DataLoader
from metrics import iou_metric
from albumentations.pytorch import ToTensorV2

from image_handler import get_bb_list, visualize
import albumentations as A

labels_to_id = {'killer': 1, 'surv': 2}


def collate_fn(batch):
    return tuple(zip(*batch))


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)*255.0
        self.std = np.array(std, dtype=np.float32)*255.0

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """

        img = tensor.cpu().numpy()
        img *= self.std
        img += self.mean

        return img


def get_transform(train):
    if train:
        transform = A.Compose([
            A.RandomCrop(width=500, height=500),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()],
            bbox_params=A.BboxParams(format='pascal_voc', min_area=512, min_visibility=0.4, label_fields=['class_labels']))
        return transform
    else:
        transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()],
            bbox_params=A.BboxParams(format='pascal_voc', min_area=512, min_visibility=0.4,
                                     label_fields=['class_labels']))
        return transform


class DbdImageDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "resize_data"))))
        self.boxes = list(sorted(os.listdir(os.path.join(root, "boxes"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "resize_data/", self.imgs[idx])
        box_path = os.path.join(self.root, "boxes/", self.boxes[idx])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bb, class_labels = get_bb_list(box_path)
        class_labels = [labels_to_id[label] for label in class_labels]
        if self.transforms is not None:
            transformed = self.transforms(image=img, bboxes=bb, class_labels=class_labels)
            img = transformed['image']
            bb = transformed['bboxes']
            class_labels = transformed['class_labels']

        # TRANSFORM TO TENSORS
        class_labels = torch.as_tensor(class_labels, dtype=torch.int64)
        bb = torch.as_tensor(bb, dtype=torch.float32)
        image_id = torch.tensor([idx])
        # area = (bb[:, 3] - bb[:, 1]) * (bb[:, 2] - bb[:, 0])

        target = {
            "boxes": bb,
            "labels": class_labels,
            "image_id": image_id,
            # "area": area
        }

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform = get_transform(train=False)
    dataset = DbdImageDataset('data/', transform)

    # image_0, target_0 = dataset[3]
    # img = image_0.permute(1, 2, 0)
    # visualize(unnorm(img), target_0['boxes'], target_0['labels'].tolist())

    data_loader = DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    for image, target in data_loader:
        images = list(image for image in image)
        targets = [{k: v for k, v in t.items()} for t in target]
        print(targets)
        image_0, target_0 = images[0], targets[0]

        image_0 = image_0.permute(1, 2, 0)
        visualize(unnorm(image_0), target_0['boxes'], target_0['labels'].tolist())
        break
