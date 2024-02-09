import pickle
import numpy as np
from matplotlib import pyplot as plt

def get_npz_data(path):
    with np.load(path, allow_pickle=True) as data_file:
        if 'gns_data' in data_file:
            data = data_file['gns_data']
        else:
            data = [item for _, item in data_file.items()]
    if len(data) > 1:
        raise NotImplementedError("Expected to have only one trajectory in npz")
    else:
        positions = data[0][0]
        runout_true = positions[-1, :, 0].max()
        friction = np.arctan(data[0][2][0]) * 180 / np.pi
        L0 = positions[0, :, 0].max() - positions[0, :, 0].min()
        H0 = positions[0, :, 1].max() - positions[0, :, 1].min()
        aspect_ratio = H0 / L0

    result = {"aspect_ratio": aspect_ratio, "friction": friction, "runout_true": runout_true}

    return result


def get_pkl_data(path):

    with open(path, "rb") as file:
        rollout_data = pickle.load(file)

    # Get data
    runout_true = rollout_data["ground_truth_rollout"][-1, :, 0].max()
    runout_pred = rollout_data["predicted_rollout"][-1, :, 0].max()
    friction = np.arctan(rollout_data["material_property"][0]) * 180 / np.pi
    L0 = rollout_data["ground_truth_rollout"][0, :, 0].max() - rollout_data["ground_truth_rollout"][0, :, 0].min()
    H0 = rollout_data["ground_truth_rollout"][0, :, 1].max() - rollout_data["ground_truth_rollout"][0, :, 1].min()
    aspect_ratio = H0 / L0

    result = {
        "aspect_ratio": aspect_ratio,
        "friction": friction,
        "runout_true": runout_true,
        "runout_pred": runout_pred
    }

    return result




a = get_pkl_data("/work2/08264/baagee/frontera/gns-mpm-data/gns-data/rollouts/sand2d_frictions-sr020/rollout_step7020000_mfmc-a069--1_ex0.pkl")







