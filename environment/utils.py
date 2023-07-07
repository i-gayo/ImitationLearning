import torch


class spatial_transform:
    """
    A class for spatial transformation for 3D image volume (batch,c,d,h,w)
    """

    def __init__(self, volsize, batch_size, device):
        """
        :param volsize: tuple (d,h,w)
        :param batch_size: the batch_size transformations apply on c volumes
        """
        self.volsize = volsize
        self.batch_size = batch_size
        self.device = device
        self.voxel_coords = torch.meshgrid(
            torch.linspace(-1, 1, volsize[2]),
            torch.linspace(-1, 1, volsize[1]),
            torch.linspace(-1, 1, volsize[0]),
            indexing="xy",
            device=self.device,
        )

    def warp(self, vol):
        """
        :param vol: 5d (batch,c,d,h,w)
        """
        return torch.nn.functional.grid_sample(vol, self.ddf)


class affine(spatial_transform):
    def __init__(self):
        super().__init__()


class spline_sparse(spatial_transform):
    def __init__(self):
        super().__init__()


class spline_grid(spatial_transform):
    def __init__(self, grid_size, **kwargs):
        super().__init__(self, **kwargs)
        """
        :param grid_size: num of control points in (d,h,w) same size between batch_size volumes
        """
        self.grid_size = grid_size
        self.control_points = torch.meshgrid(
            torch.linspace(-1, 1, grid_size[2]),
            torch.linspace(-1, 1, grid_size[1]),
            torch.linspace(-1, 1, grid_size[0]),
            indexing="xy",
            device=self.device,
        )
