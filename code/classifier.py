from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torchsummary import summary
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
negatives = []
positives = []
ROOT_PATH = "./augmentedData/"
for i in os.listdir(os.path.join(ROOT_PATH, "yes")):
    positives.append(os.path.join(os.path.join(ROOT_PATH, "yes"), i))

for i in os.listdir(os.path.join(ROOT_PATH, "no")):
    negatives.append(os.path.join(os.path.join(ROOT_PATH, "no"), i))

inputs = positives + negatives
labels = ['yes'] * len(positives) + ['no'] * len(negatives)

len(inputs), len(labels)

X_train, X_test, y_train, y_test = train_test_split(
    inputs, labels, test_size=0.33)


test_img = positives[5]
test_img = cv2.imread(test_img)
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

print(test_img.shape)
plt.imshow(test_img, cmap='gray')

yes_len = len(os.listdir(
    f"./input/yes/"))
no_len = len(os.listdir(
    f"./input/no/"))


def get_confusion_matrix(ground_truths, preds, labels):
    cm = confusion_matrix(ground_truths, preds, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()


new_size = 224

train_transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((new_size, new_size)),
    transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
    transforms.RandomRotation(degrees=(0, 180)),
    transforms.RandomPosterize(bits=2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((new_size, new_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

batch_size = 16


class BrainMRIDataset(Dataset):
    def __init__(self, files_paths, transforms=None):
        super().__init__()
        self.files_paths = files_paths
        self.transforms = transforms

    def __len__(self):
        return len(list(self.files_paths))

    def __getitem__(self, ix):
        image_path = self.files_paths[ix]
        label = image_path.split("/")[-2]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        label = 1 if label == 'yes' else 0

        return image, label

    def collate_fn(self, batch):
        imgs = []
        lbs = []
        for images, labels in batch:
            if self.transforms:
                imgs.append(self.transforms(images))
            else:
                imgs.append(images)

            lbs.append(labels)

        imgs = [torch.tensor(i) for i in imgs]
        imgs = torch.cat(imgs)
        imgs = imgs.unsqueeze(1)

        lbs = [torch.tensor(l)[None] for l in lbs]
        lbs = torch.cat(lbs)
        lbs = lbs.view(lbs.shape[0], 1)
#         lbs = F.one_hot(lbs, num_classes=2)

        return imgs.to(device), lbs.to(device)


train_ds = BrainMRIDataset(X_train, train_transformations)
test_ds = BrainMRIDataset(X_test, test_transformations)

train_dl = DataLoader(train_ds, collate_fn=train_ds.collate_fn,
                      batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_ds, collate_fn=test_ds.collate_fn,
                     batch_size=batch_size, shuffle=True)


class BrainMRIModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.dimensions = 32 * 32

        self.brain_mri_model = models.resnet18(pretrained=True)

        for param in self.brain_mri_model.parameters():
            param.requires_grad = False

        # Modify the model based on my dataset
        self.brain_mri_model.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.brain_mri_model.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.4, inplace=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(p=0.4, inplace=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.brain_mri_model(x)


def measure_accuracy(preds, labels):
    acc = ((preds > 0.5) == labels).float()
    acc = torch.sum(acc) / len(labels)

    return acc.detach().cpu()


def get_confusion_matrix(preds, labels, classes):
    # TODO: Search for methods to do this more efficiently

    y_train_, total_predictions_ = [], []

    # Convert the labels to number
    for i in labels:
        if i == 'yes':
            y_train_.append(1)
        else:
            y_train_.append(0)

    # Convert the preds to int
    for i in preds:
        x = int(i[0])
        total_predictions_.append(x)

    cm = confusion_matrix(total_predictions_, y_train_, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    plt.rcParams["figure.figsize"] = (9, 9)
    disp.plot()
    plt.show()


total_predictions = []


def train_batch(data, model, criterion, optimizer, islast=False):
    images, labels = data

    labels = labels.float()

    model.train()
    preds = model(images)

    # Measure loss
    loss = criterion(preds.float(), labels)

    # Measure accuracy
    acc = measure_accuracy(preds, labels)

    # Preprocess data for the confusion matrix
    binary_preds = ((preds > 0.5) == labels).float()

    if islast:
        total_predictions.extend(binary_preds)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.detach().item(), acc


predictions_for_cm = []


@torch.no_grad()
def test_batch(data, model, criterion):
    global predictions_for_cm

    model.eval()

    images, labels = data
    labels = labels.float()

    with torch.no_grad():
        preds = model(images)

        # Preprocess data for the confusion matrix
        binary_preds = ((preds > 0.5) == labels).float()

        predictions_for_cm.extend(binary_preds)

        loss = criterion(preds.float(), labels)
        acc = measure_accuracy(preds, labels)

        return loss.detach().item(), acc


def early_stopping(old_values, min_increment, window=3):
    """
    Returns True if it should stop earlier.
    """
    values_in_window = []

    if len(old_values) >= window:
        values_in_window = old_values[window:]
    else:
        values_in_window = old_values

    last_value = values_in_window[-1]
    avg_last_values = np.mean(values_in_window)

    if avg_last_values - last_value < min_increment:
        return True
    return False


n_epochs = 20
model = BrainMRIModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

training_loss, training_acc = [], []

is_last_epoch = False
stop = False

for ex in range(n_epochs):

    training_loss_per_epoch, training_acc_per_epoch = [], []

    print(f"Epoch {ex+1}")

    is_last_epoch = True if ex + 1 == n_epochs else False

    for ix, data in enumerate(train_dl):
        loss, acc = train_batch(data, model, criterion,
                                optimizer, is_last_epoch)
        training_loss_per_epoch.append(np.asarray(loss))
        training_acc_per_epoch.append(np.asarray(acc))

    torch.save(model.state_dict(), f"./saved_models/brain_mri_model_{ex+1}.pt")

    training_loss.append(np.asarray(training_loss_per_epoch).mean())
    training_acc.append(np.asarray(training_acc_per_epoch).mean())
    # print(f"Training loss: {training_loss[-1]}")
    print(f"Training acc: ", training_acc[-1])
