import torch


class SpatialTransform:
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
        self.compute_ddf()  # child class function
        return torch.nn.functional.grid_sample(
            vol,
            self.ddf + self.voxel_coords,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


class GlobalAffine(SpatialTransform):
    def __init__(self):
        super().__init__()


class LocalAffine(SpatialTransform):
    def __init__(self):
        super().__init__()


class GridTransform(SpatialTransform):
    def __init__(self, grid_size, interp_type="linear", **kwargs):
        super().__init__(**kwargs)
        """
        :param grid_size: num of control points in (x,y,z) same size between batch_size volumes
        """
        self.interp_type = interp_type
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
        :param rate: [0,1] *100% percentage of all control points in use
        :param scale: [0,1] scale of unit grid size the random displacement
        """
        self.control_point_displacements = (
            torch.rand(
                [
                    self.batch_size,
                    self.grid_size[1],
                    self.grid_size[0],
                    self.grid_size[2],
                    3,
                ]
            )
            * torch.tensor([self.grid_dims[i] for i in [1, 0, 2]]).reshape(
                1, 1, 1, 1, 3
            )
            * scale
            * (
                torch.rand(
                    [
                        self.batch_size,
                        self.grid_size[1],
                        self.grid_size[0],
                        self.grid_size[2],
                    ]
                )
                < rate
            )[..., None].repeat_interleave(dim=4, repeats=3)
        ).to(self.device)

    def compute_ddf(self):
        """
        Compute dense displacement field (ddf), interpolating displacement vectors on all voxels
        N.B. like all volume data, self.ddf is in y-x-z order
        """
        match self.interp_type:
            case "linear":
                self.ddf = self.linear_interpolation(
                    self.control_point_displacements, self.voxel_coords
                )
            case "g-spline_gauss":
                print("Yet implemented.")
            case "b-spline":
                print("Yet implemented.")

    @staticmethod
    def linear_interpolation(volumes, coords):
        return torch.nn.functional.grid_sample(
            input=volumes.permute(0, 4, 1, 2, 3),  # permute to (batch,c,y,x,z), c=yxz
            grid=coords,  # (batch,y,x,z,yxz)
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).permute(0, 2, 3, 4, 1)  # back to (batch,y,x,z,yxz)
