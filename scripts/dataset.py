from torch.utils import data
import random
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        
    def __len__(self):
        # Return length of the dataset
        return len(self.targets)
    
    def __getitem__(self, idx):
        # Retrieve inputs and targets at the specific index
        X = self.inputs[idx]
        y = self.targets[idx]
        return X, y

def create_datasets(snapshots, dataset_class, *args):
    """
    Create datasets for training, validation and testing
    Parameters:
    -----------------
    snapshots:      Density matrix, dim(n_train, n_atoms, n_grid, n_grid, n_grid)
    dataset_class:  Dataset class 
    args:           Format: Int
                    Arguments, splitting procentage (p_train, p_val, p_test) and 
                    number of grid points (n_grid_points)
    Return:
    -----------------
    Class object with inputs and targets as lists with numpy arrays.
    """
    I = [i for i in range(snapshots.shape[0])]

    # Partition random sizes
    num_train = random.sample(I, int(snapshots.shape[0]*args[0]))
    num_val = random.sample(I, int(snapshots.shape[0]*args[1]))
    num_test = random.sample(I, int(snapshots.shape[0]*args[2]))
    
    # Splitting the snapshots into partitions
    snapshots_train = snapshots[num_train]
    snapshots_val = snapshots[num_val]
    snapshots_test = snapshots[num_test]  
    
    def get_inputs_targets_from_snapshots(snapshots):        
        inputs = snapshots.sum(axis=1).reshape(snapshots.shape[0], 1, args[3], args[3], args[3])
        
        # Append to empty inputs and targets list, each consisting of N atoms
        # of a snapshot
        
        # The targets also contains N atoms for comparison.
        targets = []
        for i in range(snapshots.shape[0]):
            OHE = one_hot_encoding(snapshots[i,:,:], args[3]**3)
            # OHE = one_hot_encoding_diff(snapshots[i,:,:])
            OHE = OHE.reshape(args[4], args[3], args[3], args[3])
            targets.append(OHE.sum(axis=0))
        targets = np.asarray(target)
            
        return inputs, targets
    
    # Return inputs and targets for each partition
    inputs_train, targets_train = get_inputs_targets_from_snapshots(snapshots_train)
    inputs_val, targets_val = get_inputs_targets_from_snapshots(snapshots_val)
    inputs_test, targets_test = get_inputs_targets_from_snapshots(snapshots_test)
    
    # Create dataset
    training_set = dataset_class(inputs_train, targets_train)
    validation_set = dataset_class(inputs_val, targets_val)
    test_set = dataset_class(inputs_test, targets_test)

    return training_set, validation_set, test_set
