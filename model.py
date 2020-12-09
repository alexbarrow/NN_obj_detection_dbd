from data import DbdImageDataset, get_transform, collate_fn, UnNormalize
import torch
import torchvision
from eval import CocoTypeEvaluator, get_coco_api_from_dataset

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from image_handler import visualize

import time
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def init_model(classes=3):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    i = 0
    for imgs, targs in data_loader:
        # TODO: visualizing of image without boxes during train
        i += 1
        images = list(image.to(device) for image in imgs)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targs]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # TODO: losses logger
        print(losses)
        if i == 10:
            break

        # if lr_scheduler is not None:
        #     lr_scheduler.step()

        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])


@torch.no_grad()
def evaluate(model, data_loader, device):
    unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
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


def main(test_only=True):
    print('Object detection task on dbd set.')
    print('Datasets preparation...')
    transform = get_transform(train=False)

    dataset = DbdImageDataset('data/', transform)
    dataset_test = DbdImageDataset('data/val', transform)

    data_loader = DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    print('Creating model...')
    model = init_model()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #
    #                                                gamma=0.1)
    # TODO: train loop
    train_one_epoch(model, optimizer, data_loader, device=device)
    evaluate(model, data_loader_test, device=device)

    # if test_only:
    #     evaluate(model, data_loader_test, device=device)
    #     return


if __name__ == '__main__':
    main()
