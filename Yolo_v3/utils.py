# --------------------------      Intersection over Union (IoU),Non-Maximum Suppression (NMS),Mean Average Precision (mAP)    ------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def calculate_iou(predicted_boxes, target_boxes, box_format="midpoint"):
    """
    Intersection over Union (IoU) between predicted and target bounding boxes.

    predicted_boxes (torch.Tensor): Predicted bounding boxes (BATCH_SIZE, 4)
        target_boxes (torch.Tensor): Target bounding boxes (BATCH_SIZE, 4)
        box_format (str): Format of bounding boxes - "midpoint" or "corners"

    Returns:
        torch.Tensor: IoU values for all examples
    """

    if box_format == "midpoint":
        # Converting from (x, y, w, h) to (x1, y1, x2, y2)
        pred_x1 = predicted_boxes[..., 0:1] - predicted_boxes[..., 2:3] / 2
        pred_y1 = predicted_boxes[..., 1:2] - predicted_boxes[..., 3:4] / 2
        pred_x2 = predicted_boxes[..., 0:1] + predicted_boxes[..., 2:3] / 2
        pred_y2 = predicted_boxes[..., 1:2] + predicted_boxes[..., 3:4] / 2
        target_x1 = target_boxes[..., 0:1] - target_boxes[..., 2:3] / 2
        target_y1 = target_boxes[..., 1:2] - target_boxes[..., 3:4] / 2
        target_x2 = target_boxes[..., 0:1] + target_boxes[..., 2:3] / 2
        target_y2 = target_boxes[..., 1:2] + target_boxes[..., 3:4] / 2

    if box_format == "corners":
        # Boxes are already in (x1, y1, x2, y2) format
        pred_x1 = predicted_boxes[..., 0:1]
        pred_y1 = predicted_boxes[..., 1:2]
        pred_x2 = predicted_boxes[..., 2:3]
        pred_y2 = predicted_boxes[..., 3:4]  # (N, 1)
        target_x1 = target_boxes[..., 0:1]
        target_y1 = target_boxes[..., 1:2]
        target_x2 = target_boxes[..., 2:3]
        target_y2 = target_boxes[..., 3:4]

    # Calculating intersection coordinates
    intersection_x1 = torch.max(pred_x1, target_x1)
    intersection_y1 = torch.max(pred_y1, target_y1)
    intersection_x2 = torch.min(pred_x2, target_x2)
    intersection_y2 = torch.min(pred_y2, target_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection_area = (intersection_x2 - intersection_x1).clamp(0) * (intersection_y2 - intersection_y1).clamp(0)

    # Calculating box areas
    pred_area = abs((pred_x2 - pred_x1) * (pred_y2 - pred_y1))
    target_area = abs((target_x2 - target_x1) * (target_y2 - target_y1))

    union_area = pred_area + target_area - intersection_area

    return intersection_area / (union_area + 1e-6) # Adding small epsilon to avoid division by zero


def non_max_suppression(bounding_boxes, iou_threshold, confidence_threshold, box_format="corners"):
    """
    Applies Non-Maximum Suppression (NMS) to a list of bounding boxes.

    Args:
        bounding_boxes (List[List[float]]): List of bounding boxes, each specified as
            [class_prediction, confidence_score, x1, y1, x2, y2]
        iou_threshold (float): IoU threshold for considering overlapping boxes
        confidence_threshold (float): Threshold to filter low-confidence predictions
        box_format (str): Format of bounding boxes - "midpoint" or "corners"

    Returns:
        List[List[float]]: Filtered bounding boxes after applying NMS
    """

    assert isinstance(bounding_boxes, list)

    # Filtering out low-confidence predictions
    filtered_boxes = [box for box in bounding_boxes if box[1] > confidence_threshold]
    # Sorting boxes by confidence score in descending order
    sorted_boxes = sorted(filtered_boxes, key=lambda x: x[1], reverse=True)
    nms_result = []

    while sorted_boxes:
        chosen_box = sorted_boxes.pop(0)
        # Filtering out boxes with high IoU overlap
        sorted_boxes = [
            box for box in sorted_boxes
            if box[0] != chosen_box[0]
            or calculate_iou(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        nms_result.append(chosen_box)

    return nms_result


def calculate_mean_average_precision(
    predicted_boxes, ground_truth_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates Mean Average Precision (mAP) for object detection.

    Args:
        predicted_boxes (List[List[float]]): List of predicted bounding boxes, each specified as
            [image_id, class_prediction, confidence_score, x1, y1, x2, y2]
        ground_truth_boxes (List[List[float]]): List of ground truth bounding boxes, same format as predicted_boxes
        iou_threshold (float): IoU threshold for considering a prediction as correct
        box_format (str): Format of bounding boxes - "midpoint" or "corners"
        num_classes (int): Number of classes in the dataset

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    average_precisions = []

    #for numerical stability
    epsilon = 1e-6

    for class_id in range(num_classes):
        detections = []
        ground_truths = []

        for detection in predicted_boxes:
            if detection[1] == class_id:
                detections.append(detection)

        for true_box in ground_truth_boxes:
            if true_box[1] == class_id:
                ground_truths.append(true_box)

        if len(detections) > 0 and len(ground_truths) > 0:
            # print(f"\nClass {c}:")
            # print(f"Sample detection: {detections[0]}")
            # print(f"Sample ground truth: {ground_truths[0]}")

            # Debug IoU calculation
            sample_iou = calculate_iou(
                torch.tensor(detections[0][3:]),
                torch.tensor(ground_truths[0][3:]),
                box_format=box_format
            )

        if not detections or not ground_truths:
            continue

        # Counting the number of ground truth boxes for each image
        gt_counts = Counter([gt[0] for gt in ground_truths])

        for image_id, count in gt_counts.items():
            gt_counts[image_id] = torch.zeros(count)

        # Sorting detections by confidence score (descending)
        detections.sort(key=lambda x: x[2], reverse=True)
        true_positives = torch.zeros((len(detections)))
        false_positives = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):

            image_ground_truths = [
                gt for gt in ground_truths if gt[0] == detection[0]
            ]

            best_iou = 0
            best_gt_idx = None

            for idx, gt in enumerate(image_ground_truths):
                iou = calculate_iou(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if gt_counts[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    true_positives[detection_idx] = 1
                    gt_counts[detection[0]][best_gt_idx] = 1
                else:
                    false_positives[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                false_positives[detection_idx] = 1

        # Calculating cumulative sums
        TP_cumsum = torch.cumsum(true_positives, dim=0)
        FP_cumsum = torch.cumsum(false_positives, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        # Appending 0 recall and 1 precision for AUC calculation
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        # Calculating average precision using trapezoidal rule
        average_precisions.append(torch.trapz(precisions, recalls))
        
    if not average_precisions:
        print("Warning: No valid predictions were made for any class.")
        return 0.0  # Return 0 mAP if no valid predictions
    return sum(average_precisions) / len(average_precisions)


def plot_image(image, bounding_boxes, class_labels):
    # Plots predicted bounding boxes and their labels on the given image.
    image = np.array(image)
    height, width, _ = image.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    for box, label in zip(bounding_boxes, class_labels):
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"

        # Extracting box coordinates
        x_center, y_center, box_width, box_height = box
        x_min = (x_center - box_width / 2) * width
        y_min = (y_center - box_height / 2) * height

        # Creating rectangle patch
        rect = patches.Rectangle(
            (x_min, y_min),
            box_width * width,
            box_height * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Adding the rectangle to the plot
        ax.add_patch(rect)

        # Adding label above the box
        ax.text(x_min, y_min, label, color='red')

    plt.show()


def extract_bounding_boxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cpu",
):
    """
    Extracts predicted and true bounding boxes from a data loader using a given model.

    Args:
        data_loader: DataLoader providing batches of images and labels
        model: Trained model used for making predictions
        iou_threshold (float): IoU threshold for non-max suppression
        confidence_threshold (float): Confidence threshold to filter predictions
        prediction_format (str): Format of predictions - "cells" or other
        box_format (str): Format of bounding boxes - "midpoint" or "corners"
        device (str): Device to run the model on ("cpu" or "cuda")

    Returns:
        Tuple[List[List[float]], List[List[float]]]: Lists of predicted and true bounding boxes
    """
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = convert_yolo_output_to_boxes(labels)
        bboxes = convert_yolo_output_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                confidence_threshold=threshold,
                box_format=box_format,
            )


            if batch_idx == 0 and idx == 0:
                print("\nPrediction confidences before NMS:")
                for box in bboxes[idx]:
                    print(f"Class: {box[0]}, Confidence: {box[1]:.4f}")

                print("\nPredictions after NMS:")
                for box in nms_boxes:
                    print(f"Class: {box[0]}, Confidence: {box[1]:.4f}")

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_yolo_predictions(predictions, S=7):
    """
    Converts YOLO predictions from cell-relative coordinates to image-relative coordinates.

    Args:
        predictions (torch.Tensor): Raw predictions from YOLO model
        grid_size (int): Size of the grid used in YOLO (default: 7)

    Returns:
        torch.Tensor: Converted predictions in image-relative coordinates
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)

    # Extracting bounding box predictions
    box_1 = predictions[..., 21:25]
    box_2 = predictions[..., 26:30]

    # Determining which box has higher confidence score
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = box_1 * (1 - best_box) + best_box * box_2

    # Creating grid cell indices
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)

    # Converting x and y coordinates
    x_coord = 1 / S * (best_boxes[..., :1] + cell_indices)
    y_coord = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))

    # Converting width and height
    width_height = 1 / S * best_boxes[..., 2:4]

    # Combining converted coordinates
    converted_boxes = torch.cat((x_coord, y_coord, width_height), dim=-1)

     # Getting predicted class and best confidence score
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )

    # Combining all predictions
    converted_predictions = torch.cat(
        (predicted_class, best_confidence, converted_boxes), dim=-1
    )

    return converted_predictions

def convert_yolo_output_to_boxes(yolo_output, S=7):
    """
    Converts YOLO output from cell-based format to a list of bounding boxes for each image in the batch.

    Args:
        yolo_output (torch.Tensor): Raw output from YOLO model
        grid_size (int): Size of the grid used in YOLO (default: 7)

    Returns:
        List[List[List[float]]]: List of bounding boxes for each image in the batch
    """
    # Converting cell-based predictions to image-based predictions
    converted_predictions = convert_yolo_predictions(yolo_output).reshape(yolo_output.shape[0], S * S, -1)

    # Converting class predictions to long integers
    converted_predictions[..., 0] = converted_predictions[..., 0].long()
    all_bounding_boxes = []

    for image_index in range(yolo_output.shape[0]):
        image_bounding_boxes = []

        for cell_index in range(S * S):
            # Extracting and converting each prediction to a list of floats
            box = [x.item() for x in converted_predictions[image_index, cell_index, :]]
            image_bounding_boxes.append(box)
        all_bounding_boxes.append(image_bounding_boxes)

    return all_bounding_boxes

def save_checkpoint(state, filename="CS512_YOLO.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
