import numpy as np
from UNet import UNet
import densities as dm
from dataset import create_datasets
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter

def one_hot_encode(idx, input_dim):
    #Initialize empty array, dim(input_dim)
    one_hot = np.zeros(input_dim)
    
    #Set index to value 1 in one_hot
    one_hot[idx] = 1
    return one_hot
    
def one_hot_encoding(snapshot, input_dim):
    # Encode each atom coord in the snapshot
    encode = np.array([one_hot_encode(np.argmax(snapshot[i,:]), input_dim) for i in range(snapshot.shape[0])])
    
    encode = np.reshape(encode, (encode.shape[0], encode.shape[1], 1))
    return encode

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        # Moving the tensors to device
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(inputs)[:,:,:30,:30,:30]
        # Compute loss function
        loss = criterion(output, targets)
        # Compute gradients
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        train_loss /= len(data_loader)
    print('====> Epoch: {} Average loss: {:.8f}'.format(epoch, train_loss))
    return train_loss

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    for inputs, targets in test_loader:
        # Moving the tensors to device
        inputs, targets = inputs.to(device).float(), targets.to(device).long()

        output = model(inputs)[:,:,:30,:30,:30]
        loss = criterion(output, targets)
        test_loss += loss.item()

        test_loss /= len(validation_set)
    print('===> Test loss: {:.8f}'.format(test_loss))
    return test_loss

def main():
    # Load trajectory files
    traj = dm.traj_load('new.traj')
    # Convert trajectory to numpy position array
    pos_array = dm.atoms_to_array(traj, len(traj))

    # Sample 2000 random samples of all trajectory snapshots
    I = np.random.randint(0, pos_array.shape[0], pos_array.shape[0])[:2000]
    pos_array = pos_array[I]

    # Initialize parameters
    unit_cell_len = 10  # 10 Å sidelength of unit cell
    pos_array_sorted = dm.sorting_atoms_in_unitcell(pos_array, unit_cell_len)
    n_grid_points = 30
    sigma = 2 * 0.33   
    n_train = 2000  # 2000 training samples
    n_atoms = len(pos_array_sorted[1])  # Number of atoms in the system
    atomic_number = 14

    # 3D Density matrix with batch size equal to n_train
    snapshots = dm.gaussian_density_field(pos_array_sorted, 
                                        n_grid_points, sigma, n_train, 
                                        unit_cell_len, atomic_number)

    # Create dataset, and split up into training set and test set
    p_train, p_val, p_test = 0.8, 0.0, 0.2
    
    training_set, validation_set, test_set= create_datasets(snapshots, Dataset, p_train, p_val, p_test, n_grid_points)

    # Single-process data loading
    # Data fetching is performed through the same process a DataLoader is initialized
    batch_size = 10
    train_loader =  torch.utils.data.DataLoader(dataset=training_set, 
                                                batch_size=batch_size, 
                                                num_workers=0,
                                                shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=batch_size,
                                          num_workers=0,
                                          shuffle=False)

    # Initialize the network
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device("cuda" if not False and torch.cuda.is_available() else "cpu")

    net = UNet()
    model = net.to(device)

    # Defining loss function and optimizer
    learning_rate =0.0001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    NUM_EPOCHS = 100
    training_loss, validation_loss = [], []
    # Save the model for every second epoch
    for epoch in range(1, NUM_EPOCHS +1):
        training_loss.append(train(model, device, data_loader, optimizer, criterion, epoch))
        validation_loss.append(test(model, device, test_loader, criterion, epoch))
        if epoch % 2 == 0:  # save model and its parameters every second epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'BCE_tr_loss': training_loss,
                'BCE_val_loss': validation_loss
            }, "Cubic_30_Si_UNet_samples_" + str(n_train) + "_" +str(epoch)+".pt")
    
if __name__ == '__main__':
    main()
