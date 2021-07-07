from __future__ import print_function
from transforms import tensor_transform
import time
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import subprocess
import sys
import threading

import torchvision

from transforms import tensor_transform
from utils import load_data

#!pip3 install torchaudio
import torchaudio


import numpy as np
import tempfile
from scipy.io import wavfile

from microphone_stream import MicrophoneStream

import sys

import torch
from model.vit import ViT

RATE = 16000
CHUNK = int(RATE * 3.0)

if __name__ == '__main__':

    # window = [0.5] * FLAGS.avg_window

    with MicrophoneStream(RATE, CHUNK) as stream:

        audio_generator = stream.generator()
        for chunk in audio_generator:
            print("recording in progress...")

            try:
                arr = np.frombuffer(chunk, dtype=np.int16)

                #f = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir="nothing")
                wavfile.write('live_data/new.wav', RATE, arr)

            except (KeyboardInterrupt, SystemExit):
                print('Shutting Down -- closing file')

            v = ViT(
                image_size = 224,
                patch_size = 16,
                num_classes = 2,
                dim = 768,
                depth = 12,
                heads = 12,
                mlp_dim = 3072,
                dropout = 0.0,
                emb_dropout = 0.1
                )

            PATH = './model1.pth'
            net = v.to('cpu')
            net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

            waveform, sample_rate = torchaudio.load(filepath='live_data/new.wav')
            spectogram = (tensor_transform['train'])(waveform)
            spectogram = spectogram.repeat(1, 3, 1, 1)
            trans = transforms.Compose([transforms.Resize((224, 224))])
            resized = trans(spectogram)

            pred = net(resized)
            final_pred = torch.argmax(pred, dim = 1).float()
            if final_pred == 0:
                print('Child')
            else:
                print('Adult')
