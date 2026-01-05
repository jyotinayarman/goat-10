import torch
import sys
from datetime import datetime
import numpy as np
import random

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def inverse_sigmoid(x: torch.tensor) -> torch.tensor:
    return torch.log(x / (1.0 - x))


def strip_lowerdiag(L: torch.tensor) -> torch.tensor:
    # Define the indices for the lower triangular part, including the diagonal
    tril_indices = torch.tril_indices(row=3, col=3, offset=0, device=L.device)

    # Extract the lower triangular elements from each matrix in the batch
    lower_triangular_elements = L[:, tril_indices[0], tril_indices[1]]

    # Select the specific elements corresponding to the desired positions
    uncertainty = lower_triangular_elements[:, [0, 1, 2, 4, 5, 8]]

    return uncertainty


def strip_symmetric(sym: torch.tensor) -> torch.tensor:
    return strip_lowerdiag(sym)


def build_rotation_matrices(quaternions: torch.tensor) -> torch.tensor:
    # Normalize the quaternions
    quaternions = torch.nn.functional.normalize(quaternions, p=2, dim=1)

    # Extract individual components
    r, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Compute the rotation matrices
    rotation_matrices = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x ** 2 + y ** 2)
    ], dim=-1).reshape(-1, 3, 3)

    return rotation_matrices


def build_scaling_rotation(scaling_mat: torch.tensor, quaternions: torch.tensor) -> torch.tensor:
    batch_size = scaling_mat.shape[0]

    # Initialize scaling matrices L as identity matrices
    L = torch.eye(3, device=scaling_mat.device).unsqueeze(0).repeat(batch_size, 1, 1)

    # Set the diagonal elements to the scaling factors
    L *= scaling_mat.unsqueeze(2)

    # Compute rotation matrices R using the provided build_rotation function
    R = build_rotation_matrices(quaternions)

    # Perform batched matrix multiplication
    L = torch.bmm(R, L)

    return L
