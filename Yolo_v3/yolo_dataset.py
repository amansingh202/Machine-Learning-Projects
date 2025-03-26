# -------------------------------------   Loading and processing the Pascal VOC dataset.  ---------------------------------------------------

import torch
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_file: str,
        image_dir: str,
        label_dir: str,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 20,
        transform = None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index: int):
        # Loading image and bounding box data
        image, bounding_boxes = self._load_image_and_boxes(index)

        # Applying transformations if specified
        if self.transform:
            image, bounding_boxes = self.transform(image, bounding_boxes)

        # Converting bounding boxes to grid cell format
        label_matrix = self._convert_to_grid_format(bounding_boxes)

        return image, label_matrix

    def _load_image_and_boxes(self, index):
        # Loading bounding box data
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bounding_boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                bounding_boxes.append([class_label, x, y, width, height])

        # Loading the image
        img_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        return image, torch.tensor(bounding_boxes)

    def _convert_to_grid_format(self, bounding_boxes):
        label_matrix = torch.zeros((self.grid_size, self.grid_size, self.num_classes + 5 * self.num_boxes))

        for box in bounding_boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # Calculating grid cell indices
            i, j = int(self.grid_size * y), int(self.grid_size * x)

            # Calculating box coordinates relative to the cell
            x_cell, y_cell = self.grid_size * x - j, self.grid_size * y - i
            width_cell, height_cell = width * self.grid_size, height * self.grid_size

            # Assigning values to the label matrix if no object is already present in the cell
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1  # Object presence flag
                label_matrix[i, j, 21:25] = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, class_label] = 1  # One-hot encoding for class label

        return label_matrix
