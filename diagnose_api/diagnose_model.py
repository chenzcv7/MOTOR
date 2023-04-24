import torch.nn as nn
import torchvision
import torch.nn.functional as F


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, num_ftrs, num_medterm, mode='classifier'):
        super(DenseNet121, self).__init__()
        if mode == 'densenet':
            self.densenet121 = torchvision.models.densenet121(pretrained=True)
        elif mode == 'classifier':
            self.densenet121 = torchvision.models.densenet121(pretrained=True)
            self.densenet121.medterm_classifier = nn.Sequential(
                nn.Linear(num_ftrs, num_medterm),
                nn.Sigmoid()
            )

    def forward(self, input, imgs, mode='classifier'):
        if mode == 'densenet':
            features = self.densenet121.features(imgs)
            batch_feats1 = F.relu(features, inplace=True)
            out1 = F.avg_pool2d(batch_feats1, kernel_size=7, stride=1).view(batch_feats1.size(0), -1) #[1,1024]
            return out1
        else:
            medterm_probs = self.densenet121.medterm_classifier(input)
            return medterm_probs


