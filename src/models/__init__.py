"""The three intentionally lightweight, non-pretrained OFC baseline networks."""

from .cnn1d import CNN1D
from .cnn2d import CNN2D
from .stft_cnn import STFTCNN

MODELS = {"cnn1d": CNN1D, "cnn2d": CNN2D, "stft_cnn": STFTCNN}


def build_model(method, num_classes, dropout=0.3):
    return MODELS[method](num_classes=num_classes, dropout=dropout)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
