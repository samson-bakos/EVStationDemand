import os
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


class VEDAI_Dataset(torch.utils.data.Dataset):
    """Custom Dataset Class. Parses Images and Annotations for Training + Handles Edge Cases"""

    def __init__(self, annotations_dict, img_dir, transform=None):
        self.annotations_dict = annotations_dict
        self.img_dir = img_dir
        self.transform = (
            transform
            if transform is not None
            else transforms.Compose(
                [transforms.Resize((256, 256)), transforms.ToTensor()]
            )
        )
        # validate image
        self.valid_indices = [
            idx
            for idx in self.annotations_dict
            if os.path.exists(os.path.join(self.img_dir, f"{str(idx).zfill(8)}_co.png"))
            and "annotation_df" in self.annotations_dict[idx]
            and not self.annotations_dict[idx]["annotation_df"].empty
            and self.check_corners(self.annotations_dict[idx]["annotation_df"])
        ]

    @staticmethod
    def check_corners(df):
        """Check if all corners are present and valid for each image annotation"""
        required_columns = [
            "corner1_x",
            "corner1_y",
            "corner2_x",
            "corner2_y",
            "corner3_x",
            "corner3_y",
            "corner4_x",
            "corner4_y",
        ]
        if not all(column in df.columns for column in required_columns):
            return False
        return not df[required_columns].isnull().values.any()

    def validate_dataset(self):
        """Check the dataset for NaNs within images or targets."""
        for i in range(len(self.valid_indices)):
            sample = self[i]
            if sample is None:
                continue
            image, target = sample
            if torch.isnan(image).any() or any(
                torch.isnan(v).any() for v in target.values()
            ):
                print(f"NaN detected at index {self.valid_indices[i]}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        idx = self.valid_indices[index]
        img_name = os.path.join(self.img_dir, f"{str(idx).zfill(8)}_co.png")
        try:
            image = Image.open(img_name).convert("RGB")
        except (IOError, OSError) as e:
            # Skip if error
            print(f"Error loading image {img_name}: {e}")
            return None

        original_width, original_height = image.size

        # Currently resizing/ convert to tensor only
        if self.transform:
            image = self.transform(image)

        labels = self.annotations_dict[idx]["annotation_df"]
        corners = (
            labels[
                [
                    "corner1_x",
                    "corner1_y",
                    "corner2_x",
                    "corner2_y",
                    "corner3_x",
                    "corner3_y",
                    "corner4_x",
                    "corner4_y",
                ]
            ]
            .to_numpy()
            .astype(np.float32)
        )

        # rescale from 1024x1024 to 256x256
        scale_x = 256 / original_width
        scale_y = 256 / original_height
        corners[:, [0, 2, 4, 6]] *= scale_x
        corners[:, [1, 3, 5, 7]] *= scale_y

        x_min = np.min(corners[:, [0, 2, 4, 6]], axis=1)
        y_min = np.min(corners[:, [1, 3, 5, 7]], axis=1)
        x_max = np.max(corners[:, [0, 2, 4, 6]], axis=1)
        y_max = np.max(corners[:, [1, 3, 5, 7]], axis=1)

        boxes = torch.as_tensor(
            np.stack([x_min, y_min, x_max, y_max], axis=1), dtype=torch.float32
        )
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        return image, target


def train_model(model, dataset, device, epochs=10):
    """Training and Validates Model"""
    # Splitting dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Setting up DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x))
    )

    # Model to device and setting to train mode
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )

    val_losses = []  # List to store validation losses for plotting

    for epoch in range(epochs):
        running_loss = 0.0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_output = model(images, targets)
            if isinstance(loss_output, dict):
                losses = sum(loss for loss in loss_output.values())
            elif isinstance(loss_output, list):
                losses = sum(
                    loss for dict_item in loss_output for loss in dict_item.values()
                )
            else:
                raise TypeError(f"Unexpected type of loss output: {type(loss_output)}")

            losses.backward()
            optimizer.step()
            running_loss += losses.item()

        # Print average loss per epoch
        print(f"Epoch [{epoch+1}/{epochs}]")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                output_list = model(images, targets)

                for output in output_list:
                    if "scores" in output:
                        scores = output["scores"]
                        val_loss += scores.sum().item()

            # Validation loss per epoch
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f"Validation Loss: {avg_val_loss}")

        model.train()

    # Plotting the validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), val_losses, marker="o", linestyle="-", color="b")
    plt.title("Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.grid(True)
    plt.show()

    print("Training complete")
