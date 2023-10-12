import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
from datasets import PascalVOCDataset

def plot_confusion_matrix(confusion_matrix, class_list):
    plt.figure(figsize=(12, 10))

    # Use seaborn to plot the heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt="g", cmap="Blues",
                xticklabels=class_list, yticklabels=class_list)

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('D:\\paper\\Confusion Matrix-dual attention-0.1.png', format='png', dpi=300)
    plt.show()


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union
    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)
    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def get_bboxes(
        loader,
        model,
        iou_threshold,
        pred_format="cells",
        box_format="midpoint",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    all_pred_boxes = []
    all_true_boxes = []
    all_pred_labels = []
    all_true_labels = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for i, (images, boxes, labels, difficulties) in enumerate(tqdm(loader, desc='Evaluating')):
        images = images.to(device)

        with torch.no_grad():
            predicted_locs, predicted_scores = model(images)

        batch_size = images.shape[0]

        det_boxes_batch, det_labels_batch, _ = model.detect_objects(predicted_locs, predicted_scores,
                                                                    min_score=0.2, max_overlap=0.5,
                                                                    top_k=200)

        for idx in range(batch_size):

            for box in det_boxes_batch[idx]:
                all_pred_boxes.append([train_idx] + box.tolist())

            for box in boxes[idx]:
                all_true_boxes.append([train_idx] + box.tolist())

            for label in det_labels_batch[idx]:
                all_pred_labels.append([train_idx, label])

            for label in labels[idx]:
                all_true_labels.append([train_idx, label])

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes, all_pred_labels, all_true_labels


def calculate_classification_confusion_matrix(loader, model, iou_threshold, num_classes=20):
    """
    Calculate the confusion matrix for the classification part of object detection.
    """
    # Get all predicted and true bounding boxes
    all_pred_boxes, all_true_boxes, all_pred_labels, all_true_labels = get_bboxes(loader, model, iou_threshold)

    pred_classes = []
    true_classes = []

    # Handle false positives
    for true_box, true_label in zip(all_true_boxes, all_true_labels):
        max_iou = 0
        matched_class = None
        for pred_box, pred_label in zip(all_pred_boxes, all_pred_labels):
            # Check if the bounding boxes are from the same image
            if pred_box[0] == true_box[0]:
                iou = intersection_over_union(torch.tensor(pred_box[1:]), torch.tensor(true_box[1:]))
                if iou > max_iou:
                    max_iou = iou
                    matched_class = pred_label[1]  # Get the class of the true box

        # If the maximum IoU is above the threshold, consider the prediction as matched
        if max_iou > iou_threshold:
            pred_classes.append(matched_class.cpu().item())  # Get the class of the predicted box
            true_classes.append(true_label[1].cpu().item())
        else:
            # If no match is found, it's a false positive for the predicted class
            pred_classes.append(0)
            true_classes.append(true_label[1].cpu().item())  # Use an extra class index for "no match"

    # Handle false negatives
    for pred_box, pred_label in zip(all_pred_boxes, all_pred_labels):
        max_iou = 0
        for true_box, true_label in zip(all_true_boxes, all_true_labels):
            if pred_box[0] == true_box[0]:
                iou = intersection_over_union(torch.tensor([pred_box[1:]]), torch.tensor([true_box[1:]]))
                if iou > max_iou:
                    max_iou = iou

        # If no predicted box matches the ground truth box, it's a false negative
        if max_iou <= iou_threshold:
            true_classes.append(0)
            pred_classes.append(pred_label[1].cpu().item())  # Use an extra class index for "no match"

    # Calculate the confusion matrix
    cm = confusion_matrix(true_classes, pred_classes, labels=list(range(num_classes + 1)))

    return cm


def main():
    data_folder = './'
    keep_difficult = True
    batch_size = 64
    workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    test_dataset = PascalVOCDataset(data_folder,
                                    split='TEST',
                                    keep_difficult=keep_difficult)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

    checkpoint = './checkpoint_ssd300.pth.tar'

    # Load model checkpoint that is to be evaluated
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)

    class_list = [
        'no_match',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'sofa',
        'train',
        'tvmonitor',
    ]
    confusion_mat = calculate_classification_confusion_matrix(test_loader, model, iou_threshold=0.1)
    plot_confusion_matrix(confusion_mat, class_list)


if __name__ == '__main__':
    main()