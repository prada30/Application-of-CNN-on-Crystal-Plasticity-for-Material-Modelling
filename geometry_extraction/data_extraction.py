import damask
import numpy as np
import glob
import os
import h5py
import scipy.linalg as la
import time
from matplotlib import pyplot as plt

start = time.time()
def averageQuaternions(Q):
    a = np.zeros((4, 4))
    m = len(Q)
    for i in range(m):
        q = Q[i, :][..., None]
        # handle the antipodal configuration
        if q[0] < 0:
            q = np.negative(q)
        # rank 1 update
        a = np.dot(q, q.T) + a
    # scale
    a = (1.0 / m) * a
    w, vr = la.eig(a)
    # Get the eigenvector corresponding to largest eigen value
    q_avg = vr[:, np.argmax(w)]

    if q_avg[0] < 0:
        q_avg = np.negative(q_avg)
    return q_avg




path_file = os.path.join("C:\\Pappu\\Hiwi\\Task1 upload\\50 grains", "*.hdf5")
file_list = glob.glob(path_file)

number_of_files = len(file_list)

new_file = 'extracted_data.h5'
inc = 0
# initializing input_var and output_var matrices, size stems from the number of
#  files and the increments saved in every file being equal to 100


# to create input_var and output_var arrays that can store values for multiple hdf 5 files,

num_inc = 100

# counting variable
count_var = 0

for curr_file in file_list:
    # initializing data etc.
    # taking current file from list of hdf5 files
    with h5py.File(curr_file, 'r') as f:

        result = damask.Result(curr_file)
        node_coordinates = result.coordinates0_node
        cell_coordinates = result.coordinates0_point
        geometry = result.geometry0

        for key, increment in result.get(['F', 'P', 'O']).items():

            deformation_gradient = increment['F']
            stress = increment['P']/1000000.0
            orientations = increment['O']

            if key == "increment_0":
                # compute unique orientations
                _, ic, sorted_unique_counts = np.unique(orientations, axis=0, return_index=True, return_counts=True)

                # arrange unique orientations according to their first appearance in the array O
                sorted_ic = np.sort(ic)
                unique_orientations = orientations[sorted_ic]
                unique_counts = sorted_unique_counts[np.argsort(ic)]
                cell_grain_index = np.zeros((stress.shape[0], 1))

                for j_var in range(0, unique_counts.size):
                    rows = (orientations == unique_orientations[j_var, :]).all(axis=1).nonzero()
                    cell_grain_index[rows] = j_var

                index_per_grain = np.argsort(cell_grain_index, axis=0, kind='stable')

                averaged_init_orientations_per_grain = np.zeros((cell_coordinates.shape[0], 1, 4))

                initial_index = 0

                for i in range(0, unique_counts.size):
                    end_index = initial_index + unique_counts[i]
                    initial_orientations_per_grain = orientations[index_per_grain[initial_index:end_index]]
                    averaged_init_orientations_per_grain[
                        index_per_grain[initial_index: end_index]] = averageQuaternions(
                        initial_orientations_per_grain.reshape(unique_counts[i], 4))
                    initial_index = end_index


            else:

                current_deformations_per_grain = np.zeros((cell_coordinates.shape[0], 3, 3))
                current_stresses_per_grain = np.zeros((cell_coordinates.shape[0], 3, 3))
                current_orientations_per_grain = np.zeros((cell_coordinates.shape[0], 1, 4))

                initial_index = 0

                for i in range(0, unique_counts.size):
                    end_index = initial_index + unique_counts[i]

                    deformations_per_grain = deformation_gradient[
                        index_per_grain[initial_index: end_index]]
                    current_deformations_per_grain[index_per_grain[initial_index: end_index]] = np.mean(
                        deformations_per_grain, axis=0)

                    stresses_per_grain = stress[index_per_grain[initial_index: end_index]]
                    current_stresses_per_grain[index_per_grain[initial_index: end_index]] = np.mean(
                        stresses_per_grain, axis=0)

                    orientations_per_grain = orientations[
                        index_per_grain[initial_index: end_index]]
                    current_orientations_per_grain[index_per_grain[initial_index: end_index]] = averageQuaternions(
                        orientations_per_grain.reshape(unique_counts[i], 4))


                    initial_index = end_index



                with h5py.File(new_file, 'a') as file:

                    group_name = f'increment_{inc}'
                    group = file.create_group(group_name)

                    group.create_dataset('deformations_per_grain', data=current_deformations_per_grain.reshape((16,16,16,3,3)))
                    group.create_dataset('old_orientations_per_grain', data=averaged_init_orientations_per_grain.reshape((16,16,16,1,4)))
                    group.create_dataset('stresses_per_grain', data=current_stresses_per_grain.reshape((16,16,16,3,3)))
                    group.create_dataset('new_orientations_per_grain', data=current_orientations_per_grain.reshape((16,16,16,1,4)))
                    inc = inc + 1

                averaged_init_orientations_per_grain = np.copy(current_orientations_per_grain)

end = time.time()
print(end - start)
