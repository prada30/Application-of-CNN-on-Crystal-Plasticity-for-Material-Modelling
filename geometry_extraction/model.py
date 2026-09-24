import os
import time
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from torchviz import make_dot

# dummy scaler that does nothing
DummyScaler = FunctionTransformer(lambda x: x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# custom 3D CNN model
import torch
import torch.nn as nn

class Custom3DCNN(nn.Module):
    def __init__(self):
        super(Custom3DCNN, self).__init__()

        # Combined conv layers
        self.conv_combined1 = nn.Conv3d(13, 32, kernel_size=3, padding=1)
        self.batch_norm_combined1 = nn.BatchNorm3d(32)
        self.dropout1 = nn.Dropout3d(0.2)

        self.conv_combined2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.batch_norm_combined2 = nn.BatchNorm3d(32)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout3d(0.2)

        self.conv_combined3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.batch_norm_combined3 = nn.BatchNorm3d(64)
        self.max_pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout3d(0.2)

        self.conv_combined4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.batch_norm_combined4 = nn.BatchNorm3d(64)
        self.dropout4 = nn.Dropout3d(0.2)

        self.conv_combined5 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv_combined5 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.batch_norm_combined5 = nn.BatchNorm3d(32)
        self.dropout5 = nn.Dropout3d(0.2)

        self.conv_combined6 = nn.ConvTranspose3d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.conv_combined6 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.batch_norm_combined6 = nn.BatchNorm3d(32)
        self.dropout6 = nn.Dropout3d(0.2)

        self.conv_combined7 = nn.Conv3d(32, 13, kernel_size=3, padding=1)
        self.batch_norm_combined7 = nn.BatchNorm3d(13)
        self.dropout7 = nn.Dropout3d(0.2)

    def forward(self, deformation, orientation):
        # Concatenate the results along the channel dimension
        x = torch.cat((deformation, orientation), dim=1)

        # Combined conv blocks
        x = self.conv_combined1(x)
        # x = self.batch_norm_combined1(x)
        x = torch.tanh(x)
        # x = self.dropout1(x)

        x = self.conv_combined2(x)
        # x = self.batch_norm_combined2(x)
        x = torch.tanh(x)
        x = self.max_pool1(x)
        x = self.dropout2(x)

        x = self.conv_combined3(x)
        # x = self.batch_norm_combined3(x)
        x = torch.tanh(x)
        x = self.max_pool2(x)
        x = self.dropout3(x)

        x = self.conv_combined4(x)
        x = self.batch_norm_combined4(x)
        x = torch.tanh(x)
        # x = self.dropout4(x)

        x = self.conv_combined5(x)
        x = self.batch_norm_combined5(x)
        x = torch.tanh(x)
        # x = self.dropout5(x)

        x = self.conv_combined6(x)
        x = self.batch_norm_combined6(x)
        x = torch.tanh(x)
        # x = self.dropout6(x)

        x = self.conv_combined7(x)
        # x = self.batch_norm_combined7(x)
        # x = torch.tanh(x)
        # x = self.dropout7(x)

        x_stresses = x[:, :9, :, :, :]
        x_new_orientation = x[:, 9:, :, :, :]

        return x_stresses, x_new_orientation



# custom dataset class for loading data from the HDF5 file
class CustomDataset(Dataset):
    def __init__(self, hdf5_file_path, split='train', scalers=None, full_indices=None):
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')
        self.num_increments = len(self.hdf5_file.keys())
        self.split = split
        self.full_indices = self.get_full_indices(full_indices)
        self.indices = self.get_split_indices()
        self.scale_sample = 128

        if scalers is None:
            self.scalers = {
                "deformation": DummyScaler,
                "orientation": DummyScaler,
                "stresses": MinMaxScaler((-1, 1)),
                "new_orientation": DummyScaler
            }

            self.init_scalers()
        else:
            self.scalers = scalers

    def init_scalers(self):
        deformation = np.empty((0, 1), dtype=np.float64)
        orientation = np.empty((0, 1), dtype=np.float64)
        stresses = np.empty((0, 1), dtype=np.float64)
        new_orientation = np.empty((0, 1), dtype=np.float64)

        num_samples = min(self.__len__(), self.scale_sample)
        for idx in range(num_samples):
            increment = self.hdf5_file['increment_' + str(idx)]

            deformation = np.vstack((deformation , increment['deformations_per_grain'][:].reshape(-1, 1)))
            orientation = np.vstack((orientation, increment['old_orientations_per_grain'][:].reshape(-1, 1)))
            stresses = np.vstack((stresses, increment['stresses_per_grain'][:].reshape(-1, 1)))
            new_orientation = np.vstack((new_orientation, increment['new_orientations_per_grain'][:].reshape(-1, 1)))

        self.scalers['deformation'].fit(deformation)
        self.scalers['orientation'].fit(orientation)
        self.scalers['stresses'].fit(stresses)
        self.scalers['new_orientation'].fit(new_orientation)

    def get_full_indices(self, full_indices):
        if full_indices is None:
            indices = list(range(self.num_increments))

            # TODO: shuffle or no shuffle
            np.random.shuffle(indices)
            return indices
        else:
            return full_indices

    def get_split_indices(self):

        # 80% train, 10% validation, 10% test
        if self.split == 'train':
            return self.full_indices[int(0.35 * self.num_increments):]
        elif self.split == 'val':
            return self.full_indices[int(0.1 * self.num_increments):int(0.35 * self.num_increments)]
        elif self.split == 'test':
            return self.full_indices[:int(0.1 * self.num_increments)]
        else:
            raise ValueError("Invalid split. Use 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        increment = self.hdf5_file['increment_' + str(self.indices[idx])]

        # Extract cubes for deformation, orientation, stresses, and new orientation
        deformation = increment['deformations_per_grain'][:]
        orientation = increment['old_orientations_per_grain'][:]
        stresses = increment['stresses_per_grain'][:]
        new_orientation = increment['new_orientations_per_grain'][:]

        # Apply transformations using scalers
        deformation = self.scalers["deformation"].transform(deformation.reshape(-1, 1)).reshape(deformation.shape)
        orientation = self.scalers["orientation"].transform(orientation.reshape(-1, 1)).reshape(orientation.shape)
        stresses = self.scalers["stresses"].transform(stresses.reshape(-1, 1)).reshape(stresses.shape)
        new_orientation = self.scalers["new_orientation"].transform(new_orientation.reshape(-1, 1)).reshape(
            new_orientation.shape)

        # Convert NumPy arrays to PyTorch tensors
        deformation = torch.tensor(np.rollaxis(deformation.reshape((16, 16, 16, 9)), 3, 0), dtype=torch.float32,
                                   device=device)
        orientation = torch.tensor(np.rollaxis(orientation.reshape((16, 16, 16, 4)), 3, 0), dtype=torch.float32,
                                   device=device)
        stresses = torch.tensor(np.rollaxis(stresses.reshape((16, 16, 16, 9)), 3, 0), dtype=torch.float32,
                                device=device)
        new_orientation = torch.tensor(np.rollaxis(new_orientation.reshape((16, 16, 16, 4)), 3, 0), dtype=torch.float32,
                                       device=device)

        return deformation, orientation, stresses, new_orientation


# Common epoch loop for train/validation/test
# if optimizer is present, train mode, otherwise validation
def epoch_loop(model, dataloader, criterion, optimizer=None):
    losses = []

    if optimizer is not None:
        model.train()
    else:
        model.eval()

    for i, batch in enumerate(dataloader):
        deformation, orientation, stresses_gt, new_orientation_gt = batch

        # Train
        if optimizer is not None:
            # Forward pass
            stresses_pred, new_orientation_pred = model(deformation, orientation)

            # Compute loss
            pred = torch.cat((stresses_pred, new_orientation_pred), dim=1)
            true = torch.cat((stresses_gt, new_orientation_gt), dim=1)
            loss = criterion(pred, true)


            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        else:
            with torch.no_grad():
                # Forward pass
                stresses_pred, new_orientation_pred = model(deformation, orientation)

                # Compute loss
                pred = torch.cat((stresses_pred, new_orientation_pred), dim=1)
                true = torch.cat((stresses_gt, new_orientation_gt), dim=1)
                loss = criterion(pred, true)


        print(f'i {i}, Loss: {loss.item()}')

        losses.append(loss.item())

    return np.mean(losses)


def train_model(model, train_dataloader, val_dataloader, criterion, scalers, optimizer, scheduler, num_epochs):
    train_losses, val_losses, stress_rmses, orientation_maes = [], [], [], []

    for epoch in range(num_epochs):
        # Train epoch
        train_loss = epoch_loop(model, train_dataloader, criterion, optimizer)
        scheduler.step()
        train_losses.append(train_loss)

        # Validation epoch
        val_loss = epoch_loop(model, val_dataloader, criterion)
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}')

        # Validation evaluation
        test_loss, stress_rmse, orientation_mae = eval_model(model, val_dataloader,
                                                                               scalers, criterion)
        print(
            f"Val loss: {test_loss}, Stress RMSE: {stress_rmse} MAE: {orientation_mae}")
        stress_rmses.append(stress_rmse)
        orientation_maes.append(orientation_mae)

    return model, train_losses, val_losses, stress_rmses, orientation_maes


def eval_model(model, test_dataloader, scalers, criterion):
    losses = []
    stress_rmses, orientation_maes = [], []

    model.eval()

    for i, batch in enumerate(test_dataloader):
        deformation, orientation, stresses_gt, new_orientation_gt = batch

        with (torch.no_grad()):
            # Forward pass
            stresses_pred, new_orientation_pred = model(deformation, orientation)

            # Compute loss
            pred = torch.cat((stresses_pred, new_orientation_pred), dim=1)
            true = torch.cat((stresses_gt, new_orientation_gt), dim=1)
            loss = criterion(pred, true)

            # Stress RMSE
            y_pred = scalers['stresses'].inverse_transform(stresses_pred.cpu().numpy().reshape(-1, 1))
            y_true = scalers['stresses'].inverse_transform(stresses_gt.cpu().numpy().reshape(-1, 1))
            stress_rmses.append(np.sqrt(mse(y_pred, y_true)))

            # Orientation MAE
            y_pred = scalers['new_orientation'].inverse_transform(
                new_orientation_pred.cpu().numpy().reshape(-1, 1))
            y_true = scalers['new_orientation'].inverse_transform(
                new_orientation_gt.cpu().numpy().reshape(-1, 1))
            orientation_maes.append(mae(y_pred, y_true))

        losses.append(loss.item())

    return np.mean(losses), np.mean(stress_rmses), np.mean(orientation_maes)


def plot_learning_curves(train_losses, test_losses, name):
    fig = plt.gcf()
    gca = plt.gca()
    gca.plot([i for i in range(len(train_losses))], train_losses)
    gca.plot([i for i in range(len(test_losses))], test_losses)
    gca.legend(['train', 'validation'])
    gca.set_xlabel('training epochs')
    gca.set_ylabel('loss')

    plt.savefig(f"{name}.png")
    plt.show()


def plot_metrics(stress_rmses, orientation_maes, name):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    ax1.plot(range(len(stress_rmses)), stress_rmses, label='Stress RMSE')
    ax2.set_xlabel('Training epochs')
    ax1.set_ylabel('Stress RMSE')
    ax1.legend()

    ax2.plot(range(len(orientation_maes)), orientation_maes, label='Orientation MAE', color='orange')
    ax2.set_xlabel('Training epochs')
    ax2.set_ylabel('Orientation MAE')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"metrics_{name}.png")
    plt.show()


def main():
    # default data type for model, data etc.
    # torch.set_default_dtype(torch.float64)

    # Hyperparameters
    batch_size = 128
    learning_rate = 0.005
    num_epochs = 30

    # Initializes the custom datasets and dataloaders for train, val, and test
    train_dataset = CustomDataset(os.path.join(".", "data", "extracted_data.h5"), split='train')
    val_dataset = CustomDataset(os.path.join(".", "data", "extracted_data.h5"), split='val',
                                scalers=train_dataset.scalers, full_indices=train_dataset.full_indices)
    test_dataset = CustomDataset(os.path.join(".", "data", "extracted_data.h5"), split='test',
                                 scalers=train_dataset.scalers, full_indices=train_dataset.full_indices)
    print(f"data size: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # TODO: shuffle or no shuffle
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initializes the model, loss function, and optimizer
    model = Custom3DCNN()
    # model.load_state_dict(torch.load('Model\\model_dropout_convcomb2_e25_bs128_lr0.01.pth'))
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    test_loss, stress_rmse, orientation_mae = eval_model(model, test_dataloader, test_dataset.scalers,
                                                                           criterion)
    print(
        f"Test loss: {test_loss}, Stress RMSE: {stress_rmse} MAE: {orientation_mae}")

    model, train_losses, val_losses, stress_rmses, orientation_maes = train_model(model, train_dataloader,
                                                                                  val_dataloader, criterion,
                                                                                  test_dataset.scalers, optimizer,
                                                                                  scheduler, num_epochs)

    test_loss, stress_rmse, orientation_mae = eval_model(model, test_dataloader, test_dataset.scalers,
                                                                           criterion)
    print(
        f"Test loss: {test_loss}, Stress RMSE: {stress_rmse} MAE: {orientation_mae}")

    name = f"dropout_convcomb2_e{num_epochs}_bs{batch_size}_lr{learning_rate}"
    # Save the trained model if needed
    # torch.save(model.state_dict(), os.path.join(".", "model", f"model_{name}.pth"))

    plot_learning_curves(train_losses, val_losses, name)
    plot_metrics(stress_rmses, orientation_maes, name)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start)
