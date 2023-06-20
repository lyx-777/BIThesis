{
  "metadata": {
    "kernelspec": {
      "language": "python",
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "pygments_lexer": "ipython3",
      "nbconvert_exporter": "python",
      "version": "3.6.4",
      "file_extension": ".py",
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "name": "python",
      "mimetype": "text/x-python"
    },
    "colab": {
      "provenance": [],
      "include_colab_link": true
    }
  },
  "nbformat_minor": 0,
  "nbformat": 4,
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/lyx-777/BIThesis/blob/main/colab_test.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# HW3 Image Classification\n",
        "## We strongly recommend that you run with Kaggle for this homework\n",
        "https://www.kaggle.com/c/ml2022spring-hw3b/code?competitionId=34954&sortBy=dateCreated"
      ],
      "metadata": {
        "id": "jRDuJsGCgxCO"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Get Data\n",
        "Notes: if the links are dead, you can download the data directly from Kaggle and upload it to the workspace, or you can use the Kaggle API to directly download the data into colab.\n"
      ],
      "metadata": {
        "id": "EVgrPb3HhJUT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "! git init"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7qin--0niTwZ",
        "outputId": "10460fee-851f-4be5-db22-ab4d9e758c69"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Initialized empty Git repository in /content/.git/\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "ls -l\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Vos4yUjZih1M",
        "outputId": "ef30e893-2c6d-4862-bc77-4247e2cd846b"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "total 4\n",
            "drwxr-xr-x 1 root root 4096 Jun 14 18:27 \u001b[0m\u001b[01;34msample_data\u001b[0m/\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "! git branch -a"
      ],
      "metadata": {
        "id": "GMEp5h6nipwA"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#! wget https://www.dropbox.com/s/6l2vcvxl54b0b6w/food11.zip\n",
        "! wget -O food11.zip \"https://github.com/virginiakm1988/ML2022-Spring/blob/main/HW03/food11.zip?raw=true\""
      ],
      "metadata": {
        "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
        "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
        "papermill": {
          "duration": 19.351342,
          "end_time": "2022-02-23T10:03:06.247288",
          "exception": false,
          "start_time": "2022-02-23T10:02:46.895946",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:07:43.697321Z",
          "iopub.execute_input": "2022-03-03T13:07:43.697626Z",
          "iopub.status.idle": "2022-03-03T13:07:49.305932Z",
          "shell.execute_reply.started": "2022-03-03T13:07:43.697542Z",
          "shell.execute_reply": "2022-03-03T13:07:49.305076Z"
        },
        "id": "EAO6dg9eVaU_"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "! unzip food11.zip"
      ],
      "metadata": {
        "id": "HEsBm1lkhGmk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Training"
      ],
      "metadata": {
        "id": "n5ceUnRihL-f"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "_exp_name = \"sample\""
      ],
      "metadata": {
        "papermill": {
          "duration": 0.0189,
          "end_time": "2022-02-23T10:03:06.279758",
          "exception": false,
          "start_time": "2022-02-23T10:03:06.260858",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:07:49.307662Z",
          "iopub.execute_input": "2022-03-03T13:07:49.308172Z",
          "iopub.status.idle": "2022-03-03T13:07:49.313283Z",
          "shell.execute_reply.started": "2022-03-03T13:07:49.308134Z",
          "shell.execute_reply": "2022-03-03T13:07:49.311435Z"
        },
        "trusted": true,
        "id": "ay3WkYnHVaVE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Import necessary packages.\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import torch\n",
        "import os\n",
        "import torch.nn as nn\n",
        "import torchvision.transforms as transforms\n",
        "from PIL import Image\n",
        "# \"ConcatDataset\" and \"Subset\" are possibly useful when doing semi-supervised learning.\n",
        "from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset\n",
        "from torchvision.datasets import DatasetFolder, VisionDataset\n",
        "\n",
        "# This is for the progress bar.\n",
        "from tqdm.auto import tqdm\n",
        "import random"
      ],
      "metadata": {
        "papermill": {
          "duration": 1.654263,
          "end_time": "2022-02-23T10:03:07.947242",
          "exception": false,
          "start_time": "2022-02-23T10:03:06.292979",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:07:49.314913Z",
          "iopub.execute_input": "2022-03-03T13:07:49.315997Z",
          "iopub.status.idle": "2022-03-03T13:07:50.877804Z",
          "shell.execute_reply.started": "2022-03-03T13:07:49.315959Z",
          "shell.execute_reply": "2022-03-03T13:07:50.877072Z"
        },
        "trusted": true,
        "id": "CwOGtRWHVaVF"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "myseed = 6666  # set a random seed for reproducibility\n",
        "torch.backends.cudnn.deterministic = True\n",
        "torch.backends.cudnn.benchmark = False\n",
        "np.random.seed(myseed)\n",
        "torch.manual_seed(myseed)\n",
        "if torch.cuda.is_available():\n",
        "    torch.cuda.manual_seed_all(myseed)"
      ],
      "metadata": {
        "papermill": {
          "duration": 0.078771,
          "end_time": "2022-02-23T10:03:08.039428",
          "exception": false,
          "start_time": "2022-02-23T10:03:07.960657",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:07:50.879715Z",
          "iopub.execute_input": "2022-03-03T13:07:50.879924Z",
          "iopub.status.idle": "2022-03-03T13:07:50.93271Z",
          "shell.execute_reply.started": "2022-03-03T13:07:50.879894Z",
          "shell.execute_reply": "2022-03-03T13:07:50.932056Z"
        },
        "trusted": true,
        "id": "8kJm9GekVaVH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Transforms**\n",
        "Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.\n",
        "\n",
        "Please refer to PyTorch official website for details about different transforms."
      ],
      "metadata": {
        "papermill": {
          "duration": 0.01289,
          "end_time": "2022-02-23T10:03:08.065357",
          "exception": false,
          "start_time": "2022-02-23T10:03:08.052467",
          "status": "completed"
        },
        "tags": [],
        "id": "d9MVtgbSVaVH"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Normally, We don't need augmentations in testing and validation.\n",
        "# All we need here is to resize the PIL image and transform it into Tensor.\n",
        "test_tfm = transforms.Compose([\n",
        "    transforms.Resize((128, 128)),\n",
        "    transforms.ToTensor(),\n",
        "])\n",
        "\n",
        "# However, it is also possible to use augmentation in the testing phase.\n",
        "# You may use train_tfm to produce a variety of images and then test using ensemble methods\n",
        "train_tfm = transforms.Compose([\n",
        "    # Resize the image into a fixed shape (height = width = 128)\n",
        "    transforms.Resize((128, 128)),\n",
        "    # You may add some transforms here.\n",
        "    # ToTensor() should be the last one of the transforms.\n",
        "    transforms.ToTensor(),\n",
        "])\n"
      ],
      "metadata": {
        "papermill": {
          "duration": 0.021406,
          "end_time": "2022-02-23T10:03:08.099437",
          "exception": false,
          "start_time": "2022-02-23T10:03:08.078031",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:07:50.935424Z",
          "iopub.execute_input": "2022-03-03T13:07:50.935618Z",
          "iopub.status.idle": "2022-03-03T13:07:50.942922Z",
          "shell.execute_reply.started": "2022-03-03T13:07:50.935594Z",
          "shell.execute_reply": "2022-03-03T13:07:50.942191Z"
        },
        "trusted": true,
        "id": "jvI3Xmq4VaVJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "## **Datasets**\n",
        "The data is labelled by the name, so we load images and label while calling '__getitem__'"
      ],
      "metadata": {
        "papermill": {
          "duration": 0.012739,
          "end_time": "2022-02-23T10:03:08.125181",
          "exception": false,
          "start_time": "2022-02-23T10:03:08.112442",
          "status": "completed"
        },
        "tags": [],
        "id": "D0ivMf-jVaVK"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "class FoodDataset(Dataset):\n",
        "\n",
        "    def __init__(self,path,tfm=test_tfm,files = None):\n",
        "        super(FoodDataset).__init__()\n",
        "        self.path = path\n",
        "        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(\".jpg\")])\n",
        "        if files != None:\n",
        "            self.files = files\n",
        "        print(f\"One {path} sample\",self.files[0])\n",
        "        self.transform = tfm\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.files)\n",
        "\n",
        "    def __getitem__(self,idx):\n",
        "        fname = self.files[idx]\n",
        "        im = Image.open(fname)\n",
        "        im = self.transform(im)\n",
        "        #im = self.data[idx]\n",
        "        try:\n",
        "            label = int(fname.split(\"/\")[-1].split(\"_\")[0])\n",
        "        except:\n",
        "            label = -1 # test has no label\n",
        "        return im,label\n",
        "\n"
      ],
      "metadata": {
        "papermill": {
          "duration": 0.023022,
          "end_time": "2022-02-23T10:03:08.160912",
          "exception": false,
          "start_time": "2022-02-23T10:03:08.13789",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:07:50.945683Z",
          "iopub.execute_input": "2022-03-03T13:07:50.945969Z",
          "iopub.status.idle": "2022-03-03T13:07:50.954587Z",
          "shell.execute_reply.started": "2022-03-03T13:07:50.94592Z",
          "shell.execute_reply": "2022-03-03T13:07:50.953785Z"
        },
        "trusted": true,
        "id": "xBdtPhKwVaVL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class Classifier(nn.Module):\n",
        "    def __init__(self):\n",
        "        super(Classifier, self).__init__()\n",
        "        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)\n",
        "        # torch.nn.MaxPool2d(kernel_size, stride, padding)\n",
        "        # input 維度 [3, 128, 128]\n",
        "        self.cnn = nn.Sequential(\n",
        "            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]\n",
        "            nn.BatchNorm2d(64),\n",
        "            nn.ReLU(),\n",
        "            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]\n",
        "\n",
        "            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]\n",
        "            nn.BatchNorm2d(128),\n",
        "            nn.ReLU(),\n",
        "            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]\n",
        "\n",
        "            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]\n",
        "            nn.BatchNorm2d(256),\n",
        "            nn.ReLU(),\n",
        "            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]\n",
        "\n",
        "            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]\n",
        "            nn.BatchNorm2d(512),\n",
        "            nn.ReLU(),\n",
        "            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]\n",
        "\n",
        "            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]\n",
        "            nn.BatchNorm2d(512),\n",
        "            nn.ReLU(),\n",
        "            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]\n",
        "        )\n",
        "        self.fc = nn.Sequential(\n",
        "            nn.Linear(512*4*4, 1024),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(1024, 512),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(512, 11)\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        out = self.cnn(x)\n",
        "        out = out.view(out.size()[0], -1)\n",
        "        return self.fc(out)"
      ],
      "metadata": {
        "papermill": {
          "duration": 0.0258,
          "end_time": "2022-02-23T10:03:08.199437",
          "exception": false,
          "start_time": "2022-02-23T10:03:08.173637",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:07:50.955744Z",
          "iopub.execute_input": "2022-03-03T13:07:50.956055Z",
          "iopub.status.idle": "2022-03-03T13:07:50.969275Z",
          "shell.execute_reply.started": "2022-03-03T13:07:50.955996Z",
          "shell.execute_reply": "2022-03-03T13:07:50.968626Z"
        },
        "trusted": true,
        "id": "b_kDECOJVaVL"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "batch_size = 64\n",
        "_dataset_dir = \"./food11\"\n",
        "# Construct datasets.\n",
        "# The argument \"loader\" tells how torchvision reads the data.\n",
        "train_set = FoodDataset(os.path.join(_dataset_dir,\"training\"), tfm=train_tfm)\n",
        "train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)\n",
        "valid_set = FoodDataset(os.path.join(_dataset_dir,\"validation\"), tfm=test_tfm)\n",
        "valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)"
      ],
      "metadata": {
        "papermill": {
          "duration": 0.054295,
          "end_time": "2022-02-23T10:03:08.266338",
          "exception": false,
          "start_time": "2022-02-23T10:03:08.212043",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:07:50.972296Z",
          "iopub.execute_input": "2022-03-03T13:07:50.972575Z",
          "iopub.status.idle": "2022-03-03T13:07:51.01795Z",
          "shell.execute_reply.started": "2022-03-03T13:07:50.972549Z",
          "shell.execute_reply": "2022-03-03T13:07:51.017303Z"
        },
        "trusted": true,
        "id": "2_OeWtstVaVO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# \"cuda\" only when GPUs are available.\n",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "\n",
        "# The number of training epochs and patience.\n",
        "n_epochs = 4\n",
        "patience = 300 # If no improvement in 'patience' epochs, early stop\n",
        "\n",
        "# Initialize a model, and put it on the device specified.\n",
        "model = Classifier().to(device)\n",
        "\n",
        "# For the classification task, we use cross-entropy as the measurement of performance.\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "\n",
        "# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.\n",
        "optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)\n",
        "\n",
        "# Initialize trackers, these are not parameters and should not be changed\n",
        "stale = 0\n",
        "best_acc = 0\n",
        "\n",
        "for epoch in range(n_epochs):\n",
        "\n",
        "    # ---------- Training ----------\n",
        "    # Make sure the model is in train mode before training.\n",
        "    model.train()\n",
        "\n",
        "    # These are used to record information in training.\n",
        "    train_loss = []\n",
        "    train_accs = []\n",
        "\n",
        "    for batch in tqdm(train_loader):\n",
        "\n",
        "        # A batch consists of image data and corresponding labels.\n",
        "        imgs, labels = batch\n",
        "        #imgs = imgs.half()\n",
        "        #print(imgs.shape,labels.shape)\n",
        "\n",
        "        # Forward the data. (Make sure data and model are on the same device.)\n",
        "        logits = model(imgs.to(device))\n",
        "\n",
        "        # Calculate the cross-entropy loss.\n",
        "        # We don't need to apply softmax before computing cross-entropy as it is done automatically.\n",
        "        loss = criterion(logits, labels.to(device))\n",
        "\n",
        "        # Gradients stored in the parameters in the previous step should be cleared out first.\n",
        "        optimizer.zero_grad()\n",
        "\n",
        "        # Compute the gradients for parameters.\n",
        "        loss.backward()\n",
        "\n",
        "        # Clip the gradient norms for stable training.\n",
        "        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)\n",
        "\n",
        "        # Update the parameters with computed gradients.\n",
        "        optimizer.step()\n",
        "\n",
        "        # Compute the accuracy for current batch.\n",
        "        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()\n",
        "\n",
        "        # Record the loss and accuracy.\n",
        "        train_loss.append(loss.item())\n",
        "        train_accs.append(acc)\n",
        "\n",
        "    train_loss = sum(train_loss) / len(train_loss)\n",
        "    train_acc = sum(train_accs) / len(train_accs)\n",
        "\n",
        "    # Print the information.\n",
        "    print(f\"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}\")\n",
        "\n",
        "    # ---------- Validation ----------\n",
        "    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.\n",
        "    model.eval()\n",
        "\n",
        "    # These are used to record information in validation.\n",
        "    valid_loss = []\n",
        "    valid_accs = []\n",
        "\n",
        "    # Iterate the validation set by batches.\n",
        "    for batch in tqdm(valid_loader):\n",
        "\n",
        "        # A batch consists of image data and corresponding labels.\n",
        "        imgs, labels = batch\n",
        "        #imgs = imgs.half()\n",
        "\n",
        "        # We don't need gradient in validation.\n",
        "        # Using torch.no_grad() accelerates the forward process.\n",
        "        with torch.no_grad():\n",
        "            logits = model(imgs.to(device))\n",
        "\n",
        "        # We can still compute the loss (but not the gradient).\n",
        "        loss = criterion(logits, labels.to(device))\n",
        "\n",
        "        # Compute the accuracy for current batch.\n",
        "        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()\n",
        "\n",
        "        # Record the loss and accuracy.\n",
        "        valid_loss.append(loss.item())\n",
        "        valid_accs.append(acc)\n",
        "        #break\n",
        "\n",
        "    # The average loss and accuracy for entire validation set is the average of the recorded values.\n",
        "    valid_loss = sum(valid_loss) / len(valid_loss)\n",
        "    valid_acc = sum(valid_accs) / len(valid_accs)\n",
        "\n",
        "    # Print the information.\n",
        "    print(f\"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}\")\n",
        "\n",
        "\n",
        "    # update logs\n",
        "    if valid_acc > best_acc:\n",
        "        with open(f\"./{_exp_name}_log.txt\",\"a\"):\n",
        "            print(f\"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best\")\n",
        "    else:\n",
        "        with open(f\"./{_exp_name}_log.txt\",\"a\"):\n",
        "            print(f\"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}\")\n",
        "\n",
        "\n",
        "    # save models\n",
        "    if valid_acc > best_acc:\n",
        "        print(f\"Best model found at epoch {epoch}, saving model\")\n",
        "        torch.save(model.state_dict(), f\"{_exp_name}_best.ckpt\") # only save best to prevent output memory exceed error\n",
        "        best_acc = valid_acc\n",
        "        stale = 0\n",
        "    else:\n",
        "        stale += 1\n",
        "        if stale > patience:\n",
        "            print(f\"No improvment {patience} consecutive epochs, early stopping\")\n",
        "            break"
      ],
      "metadata": {
        "papermill": {
          "duration": 32830.720158,
          "end_time": "2022-02-23T19:10:19.001001",
          "exception": false,
          "start_time": "2022-02-23T10:03:08.280843",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:07:51.019126Z",
          "iopub.execute_input": "2022-03-03T13:07:51.019338Z",
          "iopub.status.idle": "2022-03-03T13:12:03.270333Z",
          "shell.execute_reply.started": "2022-03-03T13:07:51.019309Z",
          "shell.execute_reply": "2022-03-03T13:12:03.269658Z"
        },
        "trusted": true,
        "id": "zbVkfIFhVaVO"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_set = FoodDataset(os.path.join(_dataset_dir,\"test\"), tfm=test_tfm)\n",
        "test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)"
      ],
      "metadata": {
        "papermill": {
          "duration": 0.493644,
          "end_time": "2022-02-23T19:10:19.985992",
          "exception": false,
          "start_time": "2022-02-23T19:10:19.492348",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:12:03.272682Z",
          "iopub.execute_input": "2022-03-03T13:12:03.273089Z",
          "iopub.status.idle": "2022-03-03T13:12:03.288146Z",
          "shell.execute_reply.started": "2022-03-03T13:12:03.273048Z",
          "shell.execute_reply": "2022-03-03T13:12:03.287465Z"
        },
        "trusted": true,
        "id": "B9QNdHIXVaVP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Testing and generate prediction CSV"
      ],
      "metadata": {
        "papermill": {
          "duration": 0.498773,
          "end_time": "2022-02-23T19:10:20.961802",
          "exception": false,
          "start_time": "2022-02-23T19:10:20.463029",
          "status": "completed"
        },
        "tags": [],
        "id": "G31uyjpvVaVP"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model_best = Classifier().to(device)\n",
        "model_best.load_state_dict(torch.load(f\"{_exp_name}_best.ckpt\"))\n",
        "model_best.eval()\n",
        "prediction = []\n",
        "with torch.no_grad():\n",
        "    for data,_ in test_loader:\n",
        "        test_pred = model_best(data.to(device))\n",
        "        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)\n",
        "        prediction += test_label.squeeze().tolist()"
      ],
      "metadata": {
        "papermill": {
          "duration": 49.157727,
          "end_time": "2022-02-23T19:11:10.61523",
          "exception": false,
          "start_time": "2022-02-23T19:10:21.457503",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:12:03.290214Z",
          "iopub.execute_input": "2022-03-03T13:12:03.290671Z",
          "iopub.status.idle": "2022-03-03T13:12:37.20328Z",
          "shell.execute_reply.started": "2022-03-03T13:12:03.290624Z",
          "shell.execute_reply": "2022-03-03T13:12:37.202502Z"
        },
        "trusted": true,
        "id": "bpLtxx5FVaVP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#create test csv\n",
        "def pad4(i):\n",
        "    return \"0\"*(4-len(str(i)))+str(i)\n",
        "df = pd.DataFrame()\n",
        "df[\"Id\"] = [pad4(i) for i in range(1,len(test_set)+1)]\n",
        "df[\"Category\"] = prediction\n",
        "df.to_csv(\"submission.csv\",index = False)"
      ],
      "metadata": {
        "papermill": {
          "duration": 0.554276,
          "end_time": "2022-02-23T19:11:11.870035",
          "exception": false,
          "start_time": "2022-02-23T19:11:11.315759",
          "status": "completed"
        },
        "tags": [],
        "execution": {
          "iopub.status.busy": "2022-03-03T13:12:37.20452Z",
          "iopub.execute_input": "2022-03-03T13:12:37.204789Z",
          "iopub.status.idle": "2022-03-03T13:12:37.235831Z",
          "shell.execute_reply.started": "2022-03-03T13:12:37.204753Z",
          "shell.execute_reply": "2022-03-03T13:12:37.235207Z"
        },
        "trusted": true,
        "id": "fKupB3VUVaVQ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Q1. Augmentation Implementation\n",
        "## Implement augmentation by finishing train_tfm in the code with image size of your choice.\n",
        "## Directly copy the following block and paste it on GradeScope after you finish the code\n",
        "### Your train_tfm must be capable of producing 5+ different results when given an identical image multiple times.\n",
        "### Your  train_tfm in the report can be different from train_tfm in your training code.\n"
      ],
      "metadata": {
        "id": "Ivk0hrE-V8Cu"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "train_tfm = transforms.Compose([\n",
        "    # Resize the image into a fixed shape (height = width = 128)\n",
        "    transforms.Resize((128, 128)),\n",
        "    # You need to add some transforms here.\n",
        "    transforms.ToTensor(),\n",
        "])"
      ],
      "metadata": {
        "id": "GSfKNo42WjKm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Q2. Residual Implementation\n",
        "![](https://i.imgur.com/GYsq1Ap.png)\n",
        "## Directly copy the following block and paste it on GradeScope after you finish the code\n"
      ],
      "metadata": {
        "id": "3HemRgZ6WwRM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from torch import nn\n",
        "class Residual_Network(nn.Module):\n",
        "    def __init__(self):\n",
        "        super(Residual_Network, self).__init__()\n",
        "\n",
        "        self.cnn_layer1 = nn.Sequential(\n",
        "            nn.Conv2d(3, 64, 3, 1, 1),\n",
        "            nn.BatchNorm2d(64),\n",
        "        )\n",
        "\n",
        "        self.cnn_layer2 = nn.Sequential(\n",
        "            nn.Conv2d(64, 64, 3, 1, 1),\n",
        "            nn.BatchNorm2d(64),\n",
        "        )\n",
        "\n",
        "        self.cnn_layer3 = nn.Sequential(\n",
        "            nn.Conv2d(64, 128, 3, 2, 1),\n",
        "            nn.BatchNorm2d(128),\n",
        "        )\n",
        "\n",
        "        self.cnn_layer4 = nn.Sequential(\n",
        "            nn.Conv2d(128, 128, 3, 1, 1),\n",
        "            nn.BatchNorm2d(128),\n",
        "        )\n",
        "        self.cnn_layer5 = nn.Sequential(\n",
        "            nn.Conv2d(128, 256, 3, 2, 1),\n",
        "            nn.BatchNorm2d(256),\n",
        "        )\n",
        "        self.cnn_layer6 = nn.Sequential(\n",
        "            nn.Conv2d(256, 256, 3, 1, 1),\n",
        "            nn.BatchNorm2d(256),\n",
        "        )\n",
        "        self.fc_layer = nn.Sequential(\n",
        "            nn.Linear(256* 32* 32, 256),\n",
        "            nn.ReLU(),\n",
        "            nn.Linear(256, 11)\n",
        "        )\n",
        "        self.relu = nn.ReLU()\n",
        "\n",
        "    def forward(self, x):\n",
        "        # input (x): [batch_size, 3, 128, 128]\n",
        "        # output: [batch_size, 11]\n",
        "\n",
        "        # Extract features by convolutional layers.\n",
        "        x1 = self.cnn_layer1(x)\n",
        "\n",
        "        x1 = self.relu(x1)\n",
        "\n",
        "        x2 = self.cnn_layer2(x1)\n",
        "\n",
        "        x2 = self.relu(x2)\n",
        "\n",
        "        x3 = self.cnn_layer3(x2)\n",
        "\n",
        "        x3 = self.relu(x3)\n",
        "\n",
        "        x4 = self.cnn_layer4(x3)\n",
        "\n",
        "        x4 = self.relu(x4)\n",
        "\n",
        "        x5 = self.cnn_layer5(x4)\n",
        "\n",
        "        x5 = self.relu(x5)\n",
        "\n",
        "        x6 = self.cnn_layer6(x5)\n",
        "\n",
        "        x6 = self.relu(x6)\n",
        "\n",
        "        # The extracted feature map must be flatten before going to fully-connected layers.\n",
        "        xout = x6.flatten(1)\n",
        "\n",
        "        # The features are transformed by fully-connected layers to obtain the final logits.\n",
        "        xout = self.fc_layer(xout)\n",
        "        return xout"
      ],
      "metadata": {
        "id": "Q4OK9kRaWuiV"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}