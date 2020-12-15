from data import DbdImageDataset, get_transform, collate_fn, UnNormalize
import torch
import torchvision
from eval import CocoTypeEvaluator, get_coco_api_from_dataset

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from image_handler import visualize
from logger import Logger

import time
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def init_model(classes=3):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=2)
    num_classes = classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, lr_scheduler=None):
    ev_time = time.time()
    model.train()

    for imgs, targs in data_loader:
        # TODO: visualizing of image without boxes during train
        images = list(image.to(device) for image in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targs]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            # TODO: avg losses
            lr_scheduler.step(metrics=losses)

        train_time = time.time() - ev_time

        return losses.detach().item(), optimizer.param_groups[0]["lr"], train_time


@torch.no_grad()
def evaluate(model, data_loader, device):
    # unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    evaluator_time = time.time()
    model.eval()

    coco_dataset = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoTypeEvaluator(coco_dataset)

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        outputs = model(images)

        outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        # for key, img in zip(res.keys(), images):
        #     img = img.permute(1, 2, 0)
        #     visualize(unnorm(img), res[key]['boxes'].tolist(), res[key]['labels'].tolist())

        coco_evaluator.update(res)

    print('EVALUATION TIME: (t={:0.2f}s)'.format(time.time() - evaluator_time))
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    # print('RESULT')
    # print(coco_evaluator.coco_eval.stats)


def main(num_epochs):
    print('Object detection task on dbd set.')
    print('Datasets preparation...')
    transform = get_transform(train=False)

    logger = Logger()

    dataset = DbdImageDataset('data/', transform)
    dataset_test = DbdImageDataset('data/val', transform)

    data_loader = DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=2,
        collate_fn=collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=collate_fn)

    print('Creating model...')
    model = init_model()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.001, weight_decay=0.001)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,gamma=0.1)
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.75,
        patience=5,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )
    lr_sc = SchedulerClass(optimizer, **scheduler_params)

    for epoch in range(num_epochs):
        loss, lr, time = train_one_epoch(model, optimizer, data_loader, device=device, lr_scheduler=lr_sc)
        evaluate(model, data_loader_test, device=device)

        logger.update_all({'loss_total': loss, 'lr': lr, 'train_time': time}, epoch+1)
        logger.show_last()


if __name__ == '__main__':
    main(40)
