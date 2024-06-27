from flamby.datasets.fed_tcga_brca import FedTcgaBrca
from torch.utils.data import DataLoader as dl

# Load the first center as a PyTorch dataset
center0 = FedTcgaBrca(center=0, train=True)
# Load the second center as a PyTorch dataset
center1 = FedTcgaBrca(center=1, train=True)

# Sample batches from each of the local datasets using the traditional PyTorch API
X, y = iter(dl(center0, batch_size=16, shuffle=True, num_workers=0)).next()

# Print the shapes of the input and target tensors
print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")
