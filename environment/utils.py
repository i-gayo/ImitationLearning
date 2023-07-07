
import torch

class spatial_transform:
    """
    A class for spatial transformation for 3D image volume (batch,1,d,h,w)
    """
    def __init__(self, volsize, transform_type, voxdims=(1,1,1)):
        """
        :param volsize: tuple (d,h,w)
        :param voxdims: tuple (d,h,w) mm/voxel
        :param transform_type: string "affine", "spline-*"
        """
        self.volsize = volsize
        self.transform_type = transform_type
        self.voxdims = voxdims


import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineSpatialTransform3D(nn.Module):
    def __init__(self, image_size, control_points, order=3):
        """
        Initialize the B-spline spatial transformation module for 3D images.
        
        Args:
            image_size (tuple): Tuple of three integers representing the spatial dimensions of the input 3D image.
            control_points (int): Number of control points along each dimension.
            order (int, optional): B-spline order (default is 3).
        """
        super(BSplineSpatialTransform3D, self).__init__()
        self.image_size = image_size
        self.control_points = control_points
        self.order = order
        self.num_dimensions = 3
        self.num_control_points = control_points
        
        # Create the parameter grid
        self.parameter_grid = self.create_parameter_grid()
        
        # Create the B-spline basis
        self.basis = self.create_bspline_basis()
        
        # Create the transformation matrix
        self.transformation_matrix = self.create_transformation_matrix()
    
    def create_parameter_grid(self):
        """
        Create the parameter grid for B-spline basis functions.
        
        Returns:
            torch.Tensor: Parameter grid of shape (control_points, control_points, control_points, 3).
        """
        parameter_ranges = [torch.linspace(0, size - 1, control_points) for size, control_points in zip(self.image_size, self.control_points)]
        grid = torch.meshgrid(*parameter_ranges)
        grid = torch.stack(grid, dim=-1).float()
        return grid
    
    def create_bspline_basis(self):
        """
        Create the B-spline basis functions.
        
        Returns:
            torch.Tensor: B-spline basis functions of shape (control_points, control_points, control_points, 3).
        """
        basis = []
        for dim in range(self.num_dimensions):
            knots = self.parameter_grid[..., dim]
            basis_dim = F.interpolate(knots.unsqueeze(dim+3).unsqueeze(dim+3).unsqueeze(dim+3), size=self.image_size, mode='trilinear', align_corners=False)
            basis.append(basis_dim)
        basis = torch.stack(basis, dim=-1)
        return basis
    
    def create_transformation_matrix(self):
        """
        Create the transformation matrix using the B-spline basis functions.
        
        Returns:
            torch.Tensor: Transformation matrix of shape (control_points^3, image_size[0]*image_size[1]*image_size[2]).
        """
        basis_shape = self.basis.shape
        transformation_matrix = torch.zeros((self.num_control_points ** self.num_dimensions, torch.prod(torch.tensor(self.image_size))))
        
        # Reshape the basis and transformation matrix
        reshaped_basis = self.basis.view(-1, basis_shape[-1])
        reshaped_transformation_matrix = transformation_matrix.view(transformation_matrix.size(0), -1)
        
        for i in range(reshaped_basis.shape[0]):
            reshaped_transformation_matrix[i] = reshaped_basis[i].unsqueeze(1) @ reshaped_basis[i].unsqueeze(0)
        
        transformation_matrix = reshaped_transformation_matrix.view(*basis_shape[:-1], -1)
        return transformation_matrix
    
    def random_transform_parameters(self, batch_size):
        """
        Generate random transformation parameters for the spatial transformation.
        
        Args:
            batch_size (int): Size of the batch.
        
        Returns:
            tuple: Tuple containing translation, rotation, and scaling parameters.
        """
        device = self.parameter_grid.device
        control_points = self.control_points
        
        # Generate random transformation parameters
        translation = torch.randn(batch_size, self.num_dimensions).to(device) * (control_points // 4)
        rotation = torch.randn(batch_size, self.num_dimensions, self.num_dimensions).to(device) * (control_points // 8)
        scaling = torch.randn(batch_size, self.num_dimensions).to(device) * 0.2 + 1.0
        
        return translation, rotation, scaling
    
    def compute_transformed_grid(self, translation, rotation, scaling):
        """
        Compute the transformed grid using the given translation, rotation, and scaling parameters.
        
        Args:
            translation (torch.Tensor): Translation parameters of shape (batch_size, 3).
            rotation (torch.Tensor): Rotation parameters of shape (batch_size, 3, 3).
            scaling (torch.Tensor): Scaling parameters of shape (batch_size, 3).
        
        Returns:
            torch.Tensor: Transformed grid of shape (batch_size, H, W, D, 3).
        """
        batch_size = translation.size(0)
        input_size = self.image_size
        
        # Reshape the input to a flattened grid
        input_grid = F.affine_grid(torch.eye(self.num_dimensions + 1, device=translation.device).unsqueeze(0), torch.Size((batch_size, 1) + input_size))
        input_grid = input_grid.view(batch_size, -1, self.num_dimensions + 1)
        
        # Apply translation, rotation, and scaling to the input grid
        transformed_grid = input_grid[..., :self.num_dimensions]  # Initialize with original coordinates
        transformed_grid = transformed_grid - translation.unsqueeze(1)  # Apply translation
        transformed_grid = torch.einsum('bijk,bkl->bijl', transformed_grid, rotation)  # Apply rotation
        transformed_grid = transformed_grid * scaling.unsqueeze(1)  # Apply scaling
        
        return transformed_grid
    
    def forward(self, input):
        """
        Apply the B-spline based spatial transformation to the input 3D image.
        
        Args:
            input (torch.Tensor): Input 3D image tensor of shape (batch_size, 1, H, W, D).
        
        Returns:
            torch.Tensor: Transformed image tensor of shape (batch_size, 1, H, W, D).
        """
        batch_size = input.size(0)
        
        # Generate random transformation parameters
        translation, rotation, scaling = self.random_transform_parameters(batch_size)
        
        # Compute the transformed grid
        transformed_grid = self.compute_transformed_grid(translation, rotation, scaling)
        
        # Apply the grid sampler to get the transformed image
        transformed_image = F.grid_sample(input, transformed_grid, align_corners=False)
        
        return transformed_image
