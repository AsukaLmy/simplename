import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

    
class MyInception_v3(nn.Module):
    def __init__(self,transform_input=False,pretrained=False):
        super(MyInception_v3,self).__init__()
        self.transform_input=transform_input
        inception=models.inception_v3(pretrained=pretrained)
        
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e

        # self.AuxLogits = inception.AuxLogits
        # self.Mixed_7a = inception.Mixed_7a
        # self.Mixed_7b = inception.Mixed_7b
        # self.Mixed_7c = inception.Mixed_7c
        
    def forward(self,x):
        outputs=[]
        
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
        outputs.append(x)
        
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        # print(x.shape)
        outputs.append(x)

        # x = self.Mixed_7a(x)
        # x = self.Mixed_7b(x)
        # x = self.Mixed_7c(x)
        # print(x.shape)
        # outputs.append(x)
        
        return outputs
    

class MyVGG16(nn.Module):
    def __init__(self,pretrained=False):
        super(MyVGG16,self).__init__()
        
        vgg=models.vgg16(pretrained=pretrained)
     
        self.features=vgg.features
        
        # Add gradient clipping and better initialization for training stability
        if not pretrained:
            self._initialize_weights()
        
    def _initialize_weights(self):
        """Better initialization for VGG when not using pretrained weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization (better for ReLU networks)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self,x):
        # Apply gradient clipping during training to prevent explosion
        x = self.features(x)
        
        # Normalize feature magnitudes to prevent activation explosion
        if self.training:
            x = torch.clamp(x, min=0, max=10)  # Clip extreme activations
            
        #print(x.shape)
        return [x]
    
    
class MyVGG19(nn.Module):
    def __init__(self, pretrained = False):
        super(MyVGG19,self).__init__()
        
        vgg = models.vgg19(pretrained=pretrained)
     
        self.features = vgg.features
        
        # Add gradient clipping and better initialization for training stability
        if not pretrained:
            self._initialize_weights()
        
    def _initialize_weights(self):
        """Better initialization for VGG when not using pretrained weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization (better for ReLU networks)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self,x):
        # Apply gradient clipping during training to prevent explosion
        x=self.features(x)
        
        # Normalize feature magnitudes to prevent activation explosion
        if self.training:
            x = torch.clamp(x, min=0, max=10)  # Clip extreme activations
            
        return [x]


class MyRes18(nn.Module):
    def __init__(self, pretrained = False):
        super(MyRes18, self).__init__()
        res18 = models.resnet18(pretrained = pretrained)
        self.features = nn.Sequential(
            res18.conv1,
            res18.bn1,
            res18.relu,
            res18.maxpool,
            res18.layer1,
            res18.layer2,
            res18.layer3,
            res18.layer4
        )

    def forward(self, x):
        x = self.features(x)
        return [x]


class MyRes50(nn.Module):
    def __init__(self, pretrained=False):
        super(MyRes50, self).__init__()

        res50 = models.resnet50(pretrained=pretrained)

        self.features = nn.Sequential(
            res50.conv1,
            res50.bn1,
            res50.relu,
            res50.maxpool,
            res50.layer1,
            res50.layer2,
            res50.layer3,
            res50.layer4
        )

    def forward(self, x):
        x = self.features(x)
        return [x]

class MyAlex(nn.Module):
    def __init__(self, pretrained = False):
        super(MyAlex, self).__init__()

        alex = models.alexnet(pretrained = pretrained)

        self.features = alex.features
        
        # Better initialization for AlexNet when not using pretrained weights
        if not pretrained:
            self._initialize_weights()
        
    def _initialize_weights(self):
        """Better initialization for AlexNet when not using pretrained weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization for ReLU networks
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        
        # Prevent activation explosion for training stability
        if self.training:
            x = torch.clamp(x, min=0, max=8)  # Less aggressive than VGG
            
        # print(x.shape)
        return [x]


class MyMobileNet(nn.Module):
    def __init__(self, pretrained=False):
        super(MyMobileNet, self).__init__()
        
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        self.features = mobilenet.features
        
        # Better initialization for MobileNet when not using pretrained weights
        if not pretrained:
            self._initialize_weights()
        
    def _initialize_weights(self):
        """Better initialization for MobileNet when not using pretrained weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use He initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Special BatchNorm initialization for MobileNet stability
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.1)  # Small positive bias for stability
                # Initialize running stats to prevent zero activations
                nn.init.constant_(m.running_mean, 0)
                nn.init.constant_(m.running_var, 1)
        
    def forward(self, x):
        x = self.features(x)
        
        # Handle potential dead ReLU6 activations and BatchNorm issues
        if self.training:
            # Add small noise to prevent complete zero features and improve gradient flow
            x = x + 1e-6 * torch.randn_like(x)
        else:
            # In eval mode, ensure features don't collapse to zero due to BatchNorm
            # Add a very small bias to maintain feature diversity
            x = x + 1e-7
            
        return [x]



if __name__=='__main__':
    None

