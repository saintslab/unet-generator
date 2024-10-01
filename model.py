import torch
import model_specs

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size),
            torch.nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_pool_size=2):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size)
        self.max_pool = torch.nn.MaxPool2d(max_pool_size, stride=max_pool_size)
    
    def forward(self, x):
        skip_connection = self.double_conv(x)
        return self.max_pool(skip_connection), skip_connection

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, max_pool_size=2):
        super().__init__()
        #self.up = torch.nn.Upsample(scale_factor=2)
        self.up = torch.nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels * 2, out_channels, kernel_size)
        self.in_channels = in_channels
    
    def forward(self, x, skip_connection):
        x = self.up(x)

        # Crop skip connection to match x
        dx = skip_connection.shape[-2] - x.shape[-2]
        dy = skip_connection.shape[-1] - x.shape[-1]
        # if dx == 0 and dy == 0: # Assume all images are square
        #     cropped_skip = skip_connection
        # else:
        # torch._check_is_size(dx)
        # torch._check_is_size(dy)
        # torch._check(dx < skip_connection.size()[2])
        # torch._check(dy < skip_connection.size()[3])
        # cropped_skip = skip_connection[:, :, dx//2:-dx//2, dy//2:-dy//2] 

        # HACK so cropping works for perfectly fitted images
        # torch.export does not support if-else statements
        skip_connection = torch.narrow(skip_connection,2, dx//2, x.shape[-2])
        skip_connection = torch.narrow(skip_connection,3, dy//2, x.shape[-1])

        #assert cropped_skip.size()[2:] == x.size()[2:]

        x = torch.cat([x, skip_connection], dim=1)
        return self.double_conv(x)


class UNet(torch.nn.Module):
    def __init__(self, specs: model_specs.UNetSpec, initial_channels=1, final_channels=1):
        super().__init__()
        self.depth = specs.depth
        #self.quantization_level = specs.quantization_level
        self.kernel_sizes = specs.kernel_sizes
        self.initial_channels = initial_channels
        self.final_channels = final_channels

        # Defining model
        self.downs = torch.nn.ModuleList()
        self.ups = torch.nn.ModuleList()
        for i in range(self.depth):
            self.downs.append(Down(2**i * initial_channels, 2 ** (i + 1) * initial_channels, self.kernel_sizes[i]))
            self.ups.append(Up(2 ** (i+1) * initial_channels, 2 ** i * initial_channels, self.kernel_sizes[i]))
        self.middle = DoubleConv(2**(self.depth) * initial_channels, 2**(self.depth) * initial_channels, kernel_size=self.kernel_sizes[-1])
        self.final = torch.nn.Conv2d(initial_channels, final_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x, skip = down(x)
            skip_connections.append(skip)

        x = self.middle(x)

        for i in range(self.depth):
            x = self.ups[-i-1](x, skip_connections[-i - 1])
        
        return self.final(x)
        
        


