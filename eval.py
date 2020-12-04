from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from collections import defaultdict
import copy
import numpy as np
import torch
import pickle


def xyxy2xywh(bxes):
    xmin, ymin, xmax, ymax = bxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


class CocoTypeEvaluator(object):
    def __init__(self, coco_gt):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.coco_eval = COCOeval(coco_gt, iouType='bbox')

        self.img_ids = []
        self.eval_imgs = []

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.append(img_ids)

        results = self.prepare(predictions)
        coco_dt = loadRes(self.coco_gt, results)
        coco_eval = self.coco_eval

        coco_eval.cocoDt = coco_dt
        coco_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = evaluate(coco_eval)

        self.eval_imgs.append(eval_imgs)

    def prepare(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = xyxy2xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def accumulate(self):
        self.coco_eval.accumulate()

    def summarize(self):
        self.coco_eval.summarize()

    def synchronize_between_processes(self):
        # TODO: сделать список из img_ids
        self.coco_eval.evalImgs = self.eval_imgs
        self.coco_eval.params.imgIds = self.img_ids
        self.coco_eval._paramsEval = copy.deepcopy(self.coco_eval.params)
        print(self.coco_eval._paramsEval.areaRng)
        print(self.coco_eval._paramsEval.imgIds)


def createIndex(self):
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if 'annotations' in self.dataset:
        for ann in self.dataset['annotations']:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

    if 'images' in self.dataset:
        for img in self.dataset['images']:
            imgs[img['id']] = img

    if 'categories' in self.dataset:
        for cat in self.dataset['categories']:
            cats[cat['id']] = cat

    if 'annotations' in self.dataset and 'categories' in self.dataset:
        for ann in self.dataset['annotations']:
            catToImgs[ann['category_id']].append(ann['image_id'])

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


def loadRes(self, resFile):
    """
    Load result file and return a result api object.
    :param   resFile (str)     : file name of result file
    :return: res (obj)         : result api object
    """
    res = COCO()
    res.dataset['images'] = [img for img in self.dataset['images']]

    # print('Loading and preparing results...')
    # tic = time.time()
    anns = resFile
    assert type(anns) == list, 'results in not an array of objects'

    annsImgIds = [ann['image_id'] for ann in anns]

    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
        'Results do not correspond to current coco set'

    if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])

        for id, ann in enumerate(anns):
            bb = ann['bbox']
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0

    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset['annotations'] = anns
    createIndex(res)
    return res


def evaluate(self):
    '''
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    '''
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    p.iouType = 'bbox'
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    computeIoU = self.computeIoU

    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]

    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]
    # print(evalImgs)
    # this is NOT in the pycocotools code, but could be done outside
    # evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    # self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


def get_coco_api_from_dataset(ds):
    coco_ds = COCO()
    ann_id = 1
    dataset = {'images': [], 'categories': [], 'annotations': []}
    categories = set()

    for img_idx in range(len(ds)):
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()

        img_dict = {}
        img_dict['id'] = image_id
        img_dict['height'] = img.shape[-2]
        img_dict['width'] = img.shape[-1]
        dataset['images'].append(img_dict)

        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets['labels'].tolist()
        areas = targets['area'].tolist()
        iscrowd = targets['iscrowd'].tolist()

        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann['image_id'] = image_id
            ann['bbox'] = bboxes[i]
            ann['category_id'] = labels[i]
            categories.add(labels[i])
            ann['area'] = areas[i]
            ann['id'] = ann_id
            ann['iscrowd'] = iscrowd[i]

            dataset['annotations'].append(ann)
            ann_id += 1

    dataset['categories'] = [{'id': i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


if __name__ == '__main__':
    # coco_dataset = get_coco_api_from_dataset(data_loader.dataset)
    # coco_evaluator = CocoTypeEvaluator(coco_dataset)
    print('Ready!')