from torchvision import transforms
import torchaudio
import torch

tensor_transform = {
    'train':
        transforms.Compose([
            torchaudio.transforms.MelSpectrogram(),
            #transforms.Resize(size=(768, 768))
            # transforms.ToTensor(),  # Imagenet standards
        ]),
    'valid':
        transforms.Compose([
            torchaudio.transforms.MelSpectrogram(),
            # transforms.ToTensor(),
        ]),
}