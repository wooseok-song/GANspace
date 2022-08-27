import torch
import torchvision
import matplotlib.pyplot as plt


############################## Visualize image using make_grid function || Input => BCHW
def visualize_grid(image):

    grid = torchvision.utils.make_grid(image.detach().cpu(), normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()



