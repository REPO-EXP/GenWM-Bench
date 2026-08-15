from . import *
from .noise_layers import *

class Random_Noise(nn.Module):

    def __init__(self, layers, len_layers_R, len_layers_F):
        super(Random_Noise, self).__init__()
        for i in range(len(layers)):
            layers[i] = eval(layers[i])
        self.noise = nn.Sequential(*layers)
        self.len_layers_R = len_layers_R
        self.len_layers_F = len_layers_F
        print(self.noise)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def forward(self, image_cover_mask):
        image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]
        forward_image = image.clone().detach()
        forward_cover_image = cover_image.clone().detach()
        forward_mask = mask.clone().detach()
        noised_image_C = torch.zeros_like(forward_image)
        noised_image_R = torch.zeros_like(forward_image)
        noised_image_F = torch.zeros_like(forward_image)

        for index in range(forward_image.shape[0]):
            random_noise_layer_C = np.random.choice(self.noise, 1)[0]
            random_noise_layer_R = np.random.choice(self.noise[0:self.len_layers_R], 1)[0]
            random_noise_layer_F = np.random.choice(self.noise[self.len_layers_R:self.len_layers_R + self.len_layers_F], 1)[0]
            noised_image_C[index] = random_noise_layer_C([forward_image[index].clone().unsqueeze(0), forward_cover_image[index].clone().unsqueeze(0), forward_mask[index].clone().unsqueeze(0)])
            noised_image_R[index] = random_noise_layer_R([forward_image[index].clone().unsqueeze(0), forward_cover_image[index].clone().unsqueeze(0), forward_mask[index].clone().unsqueeze(0)])
            noised_image_F[index] = random_noise_layer_F([forward_image[index].clone().unsqueeze(0), forward_cover_image[index].clone().unsqueeze(0), forward_mask[index].clone().unsqueeze(0)])

        noised_image_gap_C = noised_image_C.clamp(-1, 1) - forward_image
        noised_image_gap_R = noised_image_R.clamp(-1, 1) - forward_image
        noised_image_gap_F = noised_image_F.clamp(-1, 1) - forward_image

        return image + noised_image_gap_C, image + noised_image_gap_R, image + noised_image_gap_F
