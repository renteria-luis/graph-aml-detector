'''Quick environment check for the AML graph detector.

Confirms PyTorch sees the GPU, that PyG and its sampling backend are
importable, and that a SAGEConv layer actually runs on CUDA. The
NeighborLoader check matters because the whole project relies on
neighbor sampling, which needs pyg-lib or torch-sparse underneath.
'''

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv


def main() -> None:
    print('torch', torch.__version__)
    print('cuda build', torch.version.cuda)
    print('cuda available', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('device', torch.cuda.get_device_name(0))

    # Confirm the sampling backend is present. If both imports fail,
    # NeighborLoader will break later, so we want the signal now.
    try:
        import pyg_lib  # noqa: F401
        print('pyg_lib', 'ok')
    except ImportError:
        print('pyg_lib', 'missing')
    try:
        import torch_sparse  # noqa: F401
        print('torch_sparse', 'ok')
    except ImportError:
        print('torch_sparse', 'missing')

    # One GraphSAGE layer forward pass on the GPU.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    x = torch.randn(4, 8)
    conv = SAGEConv(8, 16).to(device)
    out = conv(x.to(device), edge_index.to(device))
    print('sageconv output', tuple(out.shape), out.device.type)

    # Neighbor sampling on a CPU graph. The sampler runs on CPU and hands
    # batches to the model, so keeping the source data on CPU here avoids
    # device-mismatch surprises.
    data = Data(x=x, edge_index=edge_index)
    loader = NeighborLoader(data, num_neighbors=[2, 2], batch_size=2)
    batch = next(iter(loader))
    print('neighborloader batch nodes', batch.num_nodes)

    print('environment looks good')


if __name__ == '__main__':
    main()
