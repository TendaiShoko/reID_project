import torch.nn as nn
from safetensors import torch

from .foundation_models import FeatureExtractor


class KnowledgeDistillationModel(nn.Module):
    def __init__(self, teacher_model, num_classes):
        super(KnowledgeDistillationModel, self).__init__()
        self.teacher = teacher_model
        self.student = FeatureExtractor('resnet50', pretrained=False)
        self.student_fc = nn.Linear(self.student.feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            teacher_output = self.teacher(x)
        student_features = self.student(x)
        student_output = self.student_fc(student_features)
        return teacher_output, student_output
