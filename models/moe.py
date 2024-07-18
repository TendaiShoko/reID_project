import torch
import torch.nn as nn
from .foundation_models import FeatureExtractor


class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, num_classes, model_names=['resnet50', 'vit'], common_dim=512):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([FeatureExtractor(name) for name in model_names])
        self.common_dim = common_dim

        # Adding a linear layer to project features to the common dimension
        self.projection_layers = nn.ModuleList(
            [nn.Linear(expert.feature_dim, self.common_dim) for expert in self.experts])
        self.gating = nn.Linear(self.common_dim, num_experts)
        self.fc = nn.Linear(self.common_dim, num_classes)

    def forward(self, x):
        expert_outputs = [proj(expert(x)) for proj, expert in zip(self.projection_layers, self.experts)]
        gating_weights = torch.softmax(self.gating(expert_outputs[0]), dim=1)
        combined_output = sum(gating_weight.unsqueeze(1) * expert_output for gating_weight, expert_output in
                              zip(gating_weights.t(), expert_outputs))
        return self.fc(combined_output)
