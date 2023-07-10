import torch


class spatial_transform:
    """
    A class for spatial transformation for 3D image volume (batch,c,y,x,z)
    """

    def __init__(self, volsize, batch_size, device):
        """
        :param volsize: tuple (x,y,z)
        :param batch_size: the batch_size transformations apply on c volumes
        """
        self.volsize = volsize
        self.batch_size = batch_size
        self.device = device
        self.voxel_coords = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, volsize[1]),
                    torch.linspace(-1, 1, volsize[0]),
                    torch.linspace(-1, 1, volsize[2]),
                    indexing="ij",
                ),
                dim=3,
            )[None, ...]
            .repeat_interleave(dim=0, repeats=self.batch_size)
            .to(self.device)
        )

    def warp(self, vol):
        """
        :param vol: 5d (batch,c,y,x,z)
        """
        return torch.nn.functional.grid_sample(vol, self.ddf)


class global_affine(spatial_transform):
    def __init__(self):
        super().__init__()


class local_affine(spatial_transform):
    def __init__(self):
        super().__init__()


class grid_transform(spatial_transform):
    def __init__(self, grid_size, **kwargs):
        super().__init__(**kwargs)
        """
        :param grid_size: num of control points in (x,y,z) same size between batch_size volumes
        """
        self.grid_size = grid_size
        self.control_point_coords = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, grid_size[1]),
                    torch.linspace(-1, 1, grid_size[0]),
                    torch.linspace(-1, 1, grid_size[2]),
                    indexing="ij",
                ),
                dim=3,
            )[None, ...]
            .repeat_interleave(dim=0, repeats=self.batch_size)
            .to(self.device)
        )
        self.grid_dims = [2 / (self.grid_size[i] - 1) for i in [0, 1, 2]]  # (x,y,z)

        self.control_point_displacements = torch.zeros_like(self.control_point_coords)

    def generate_random_transform(self, rate=0.25, scale=0.1):
        """
        Generate random displacements on control points dcp (uniform distribution)
        :param rate: [0,1] x100% percentage of all control points in use
        :param scale: [0,1] scale of unit grid size the random displacement
        """
        self.control_point_displacements = (
            torch.rand([self.batch_size, self.grid_size[1], self.grid_size[0], self.grid_size[2], 3])
            * torch.tensor(self.grid_dims).reshape(1, 1, 1, 1, 3)
            * scale
            * (torch.rand([self.batch_size, self.grid_size[1], self.grid_size[0], self.grid_size[2]]) < rate)[
                ..., None
            ].repeat_interleave(dim=4, repeats=3)
        )

    def compute_ddf(self):
        return 0
