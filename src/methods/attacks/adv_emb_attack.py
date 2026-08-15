import torch
import torch.nn as nn
import gc 
from torchvision import transforms
from PIL import Image

from src.core.interfaces import BaseAttack
from src.core.registry import ATTACKS
from src.methods.attacks.feature_extractors import ResNet18Embedding, ClipEmbedding, VAEEmbedding,KLVAEEmbedding

class WarmupPGDEmbedding:
    def __init__(
        self,
        model,
        device,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        loss_type="l2",
        random_start=True,
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss_type = loss_type
        self.random_start = random_start
        self.device = device

        if self.loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif self.loss_type == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")

    def forward(self, images):
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        
        images = images.clone().detach().to(self.device)
        original_embeddings = self.model(images).detach()

        if self.random_start:
            adv_images = images.clone().detach()
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            adv_embeddings = self.model(adv_images)

            cost = self.loss_fn(adv_embeddings, original_embeddings)

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

@ATTACKS.register("AdvEmbAttack")
class AdvEmbAttack(BaseAttack):
    def __init__(self, encoder="resnet18", strength=2.0, steps=20, device='cuda'):
        super().__init__()
        self.encoder_name = encoder
        self.strength = float(strength)
        self.steps = int(steps)
        self.device = device
        
        self.eps_factor = 1 / 255
        self.alpha_factor = 0.05
        
        self.embedding_model = self._load_model()
        self.attacker = self._setup_attack()
        
        self.transform = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def _load_model(self):
        print(f"   [AdvEmb] Loading encoder: {self.encoder_name}...")
        
        model = None
        if self.encoder_name == "resnet18":
            model = ResNet18Embedding()
        elif self.encoder_name == "clip":
            model = ClipEmbedding()
        elif self.encoder_name == "klvae8": 
            model = VAEEmbedding("stabilityai/sd-vae-ft-mse")
        elif self.encoder_name == "klvae16": 
            model = KLVAEEmbedding("kl-f16")
        elif self.encoder_name == "sdxlvae": 
            model = VAEEmbedding("stabilityai/sdxl-vae")
        else:
            raise ValueError(f"Unsupported encoder: {self.encoder_name}")
        
        return model.to(self.device)

    def _setup_attack(self):
        return WarmupPGDEmbedding(
            model=self.embedding_model,
            device=self.device,
            eps=self.eps_factor * self.strength,
            alpha=self.alpha_factor * self.eps_factor * self.strength,
            steps=self.steps,
            loss_type="l2"
        )

    def apply(self, image: Image.Image) -> Image.Image:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        try:

            if self.embedding_model is None:
                 self.embedding_model = self._load_model()
                 self.attacker.model = self.embedding_model 

            adv_tensor = self.attacker.forward(img_tensor)
            
            res_tensor = adv_tensor.squeeze(0).cpu()
            adv_pil = self.to_pil(res_tensor)
            
            return adv_pil

        finally:

            del img_tensor
            if 'adv_tensor' in locals(): del adv_tensor
            if 'res_tensor' in locals(): del res_tensor

    def unload(self):
        if self.embedding_model is not None:
            print(f"   [AdvEmb] Unloading encoder: {self.encoder_name} to free VRAM.")
            del self.embedding_model
            self.embedding_model = None
            if self.attacker:
                self.attacker.model = None
            gc.collect()
            torch.cuda.empty_cache()

    def get_param_str(self):
        return f"AdvEmb_{self.encoder_name}_s{int(self.strength)}"