from torchvision.models import resnet50
from Vit import vit_encoder_b, load_weight_for_vit_encoder
from torchvision import transforms
import torch.nn as nn
import torch

import PIL


def load_model(cond_ckpt_path):
    model = resnet50()
    modules = list(model.children())[:-2]
    model = torch.nn.Sequential(*modules)
    for name, _ in model.named_modules():
        print(name)
    model.fc = nn.Sequential(
        nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(128, 4, kernel_size=1, stride=1, padding=0),
    )

    # num_ftrs = model.fc.in_features
    # model.fc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 4))
    pretrained_weights = torch.load(cond_ckpt_path)
    model.load_state_dict(pretrained_weights, strict=False)
    print(f"Loaded weights from {cond_ckpt_path}")
    model.eval()
    return model


def load_vit_model(cond_ckpt_path):
    model = vit_encoder_b()
    model.fc = nn.Sequential(
        nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(128, 4, kernel_size=1, stride=1, padding=0),
    )
    pretrained_weights = load_weight_for_vit_encoder("lvm-med-vit", cond_ckpt_path)
    model.load_state_dict(pretrained_weights, strict=False)
    model.eval()
    return model


if __name__ == "__main__":
    cond_ckpt_path = "/disk/cyq/2024/My_Proj/Xray-Diffusion/ldm/Medicalnet/lvmmed_resnet.torch"
    vit_cond = "/disk/cyq/2024/My_Proj/Xray-Diffusion/ldm/Medicalnet/lvmmed_vit.pth"
    test_image_path = "/disk/ssy/data/drr/result/split/zhouguDR/01/01_01.jpg"

    model = load_vit_model(vit_cond)

    data_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    # * PIL load image
    img = PIL.Image.open(test_image_path)
    img_rgb = img.convert("RGB")

    img_rgb.save("./rgb.jpg")
    img = data_transforms(img_rgb)
    img = img.unsqueeze(0)
    out = model(img)
    print(out.shape)
