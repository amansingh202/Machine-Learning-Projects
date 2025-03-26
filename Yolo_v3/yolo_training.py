# ---------------------------       Main script for training YOLO model on Pascal VOC dataset    -----------------------------
import argparse
import sys
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import random
from typing import List

from utils import calculate_mean_average_precision, convert_yolo_output_to_boxes, extract_bounding_boxes, load_checkpoint, non_max_suppression, plot_image, save_checkpoint
from yolo_dataset import VOCDataset
from yolo_loss import YoloLoss
from yolo_model import YoloVersion1

#random seed for reproducibility
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)


# Pascal VOC dataset classes
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


load_trained_model = 0

# Training configurations
CONFIG = {
    "LEARNING_RATE": 2e-5,
    "DEVICE": "cpu",  # we have used "cpu" only
    "BATCH_SIZE": 16,
    "WEIGHT_DECAY": 0,
    "EPOCHS": 100,
    "NUM_WORKERS": 2,
    "PIN_MEMORY": True,
    "LOAD_MODEL": False,
    "LOAD_MODEL_FILE": "CS512_YOLO_577.pth.tar",
    "IMG_DIR": "PASCAL_VOC/images",
    "LABEL_DIR": "PASCAL_VOC1/labels",
}
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

# Defining image transformations
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_epoch(train_loader, model, optimizer, loss_fn):
    """
    Trains the model for one epoch
    """
    model.train()
    progress_bar = tqdm(train_loader, leave=True)
    epoch_losses = []

    for batch_idx, (images, targets) in enumerate(progress_bar):
        images, targets = images.to(CONFIG["DEVICE"]), targets.to(CONFIG["DEVICE"])

        predictions = model(images)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        progress_bar.set_postfix(loss=loss.item())

    mean_loss = sum(epoch_losses) / len(epoch_losses)
    print(f"Mean loss for this epoch: {mean_loss:.4f}")


def main():  
    # Initializing model, optimizer, and loss function
    model = YoloVersion1(split_size=7, num_boxes=2, num_classes=20).to(CONFIG["DEVICE"])
    optimizer = optim.Adam(
        model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=CONFIG["WEIGHT_DECAY"]
    )
    loss_fn = YoloLoss()

    if CONFIG["LOAD_MODEL"]:
        load_checkpoint(torch.load(CONFIG["LOAD_MODEL_FILE"]), model, optimizer)

    # Loading and preparing datasets
    train_dataset = VOCDataset(
        "PASCAL_VOC//train.csv",
        transform=transform,
        image_dir=CONFIG["IMG_DIR"],
        label_dir=CONFIG["LABEL_DIR"],
    )

    # subset of the training data (1000 samples)
    # train_subset = Subset(train_dataset, random.sample(range(len(train_dataset)), 1000))

    test_dataset = VOCDataset(
        "PASCAL_VOC//test.csv",
        transform=transform,
        image_dir=CONFIG["IMG_DIR"],
        label_dir=CONFIG["LABEL_DIR"],
    )

    # Data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=CONFIG["PIN_MEMORY"],
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        num_workers=CONFIG["NUM_WORKERS"],
        pin_memory=CONFIG["PIN_MEMORY"],
        shuffle=True,
        drop_last=True,
    )

    # Training loop
    for epoch in range(CONFIG["EPOCHS"]):
        #################################################################
        # this code is required if we want to view our model performance on some of the images
        if CONFIG["LOAD_MODEL"]:
            for x, y in train_loader:
                x = x.to(CONFIG["DEVICE"])
                for idx in range(8):
                    bboxes = convert_yolo_output_to_boxes(model(x))
                    bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, confidence_threshold=0.4, box_format="midpoint")
                    #plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

                    # Extract class predictions
                    class_preds = [box[0] for box in bboxes]

                    # Convert class indices to labels (assuming you have a list of class names)
                    class_labels = [VOC_CLASSES[int(pred)] for pred in class_preds]

                    # Plot image with bounding boxes and class labels
                    plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes, class_labels)

            import sys
            sys.exit()
        ##############################################################

        train_epoch(train_loader, model, optimizer, loss_fn)

        # Evaluating the model
        pred_boxes, target_boxes = extract_bounding_boxes(
            train_loader, model, iou_threshold=0.1, threshold=0.05
        )

        # Calculation mAP
        mean_avg_prec = calculate_mean_average_precision(
            pred_boxes,
            target_boxes,
            iou_threshold=0.1,
            box_format="midpoint"
        )
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - Train mAP: {mean_avg_prec:.4f}")


        # Save checkpoint after each epoch (optional)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=CONFIG["LOAD_MODEL_FILE"])


if __name__ == "__main__":
    main()
