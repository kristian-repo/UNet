from ase.io import Trajectory
import numpy as np

def traj_load(filename):
    """
    Load trajectory file with ase 
    Parameter:
    ------------
    filename, str (".traj")
    
    Return: List of ASE objects
    """
    atoms_object = Trajectory(filename)
    return atoms_object

def atoms_to_array(atoms_object, n_train):
    """
     Takes an ASE atom object and converts it to a matrix of size
    (n_train, n_atoms, 3). n_atoms represents the number of atoms
    in the system. Each row represents an atom and each column
    is the xyz coordinates
    
    Parameter: 
    --------------
    atoms_object: ASE object
    n_train: Number of training samples
    Return: numpy array, dim(n_train, n_atoms, 3)
    """
    pos = []
    for i in range(n_train):
        atoms = atoms_object[i]
        xyz = atoms.get_positions()
        pos.append(xyz)
    return np.array(pos)


def sorting_atoms_in_unitcell(pos, unit_cell_len):
    """
    Sorts the atomic positions within a defined unit cell
    
    Paramter:
    --------------
    pos: Position numpy array, dim(n_train, n_atoms, 3)
    unit_cell_len: int, side length of unit cell
    Return: Numpy array with sorted atomic positions
    """

    n_atoms = pos.shape[1]
    pos_seq = []

    # A simple cubic unit cell that consists of 8 corners
    unitcell_corners = np.array([[0, 0, 0], [unit_cell_len, 0, 0], [0, unit_cell_len, 0],
                                 [0, 0, unit_cell_len], [unit_cell_len, unit_cell_len, 0], [unit_cell_len, 0, unit_cell_len],
                                 [0, unit_cell_len, unit_cell_len], [unit_cell_len, unit_cell_len, unit_cell_len]])
    
    for i in range(len(pos)):
        xyz = pos[i]

        dist_vec = np.zeros((n_atoms, 8))  
        for j in range(8):
            dist_vec[:,j] = np.linalg.norm(xyz - unitcell_corners[j], axis=1) # l2-norm over each xyz-coordinate of each atom

        dist = np.min(dist_vec, axis=1) # identifying the shortest distance to origin of the 8 corners
        pos_sorted = xyz[dist.argsort()]   # Indexing the minimum distance to each atomic position
        pos_seq.append(pos_sorted)

    return np.array(pos_seq)


def gaussian_density_field(input_array, n_grid_points, sigma, n_train, unit_cell_len, atomic_number):
    """
    Calculates a 3D density field of a cubic unit cell
    Parameter:
    ------------
    input_array, dim(n_train, n_atoms, 3)
    n_grid_points: Number of grid points/voxels for each cartesian coordinate direction
    sigma: Width of Gaussian distribution (or kernel parameter for density smoothing)
    n_train: Number of training examples 
    unit_cell_len: Side length of cubic unit cell
    Returns:
    ----------
    density_matrix, dim(n_train, n_atoms, n_grid_points**3)
    """

    # Grid in 1D
    grid = np.linspace(0, unit_cell_len, n_grid_points + 1)
    d_spaces = grid[:-1] + (grid[1]-grid[0])/2

    pos_matrix = input_array[0:n_train]
    n_atoms = len(pos_matrix[0])

    num = 0
    # Construct empty 3D grid box
    grid_3d = np.zeros((n_grid_points**3, 3))
    for i in range(n_grid_points):
        for j in range(n_grid_points):
            for k in range(n_grid_points):
                grid_3d[num, :] = np.array([d_spaces[i], d_spaces[j], d_spaces[k]])
                num += 1

    density_matrix = np.zeros((n_train, n_atoms), object)
    for i in range(n_train):
        for j in range(n_atoms):
            density_matrix[i][j] = 1/((sigma*np.sqrt(2*np.pi))**3)*atomic_number*np.exp(-np.linalg.norm(grid_3d - pos_matrix[i][j], axis=1)**2/(2*sigma**2))

    density_matrix = np.asarray(density_matrix.tolist())  # convert list to numpy array
    return density_matrix
