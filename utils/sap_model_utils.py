import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
import torch 

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin[0]+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/float(stride[0])+1)), int(np.floor((Lin[1]+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/float(stride[1])+1))

def reshape_conv_input_activation(x, conv_layer=None, kernel_size=3, stride=1, padding=0, dilation=1):
    ### FAST CODE (Avoid for loops)
    if conv_layer:
        kernel_size = conv_layer.kernel_size
        stride = conv_layer.stride
        padding =  conv_layer.padding 
        dilation = conv_layer.dilation
    x_unfold = torch.nn.functional.unfold(x, kernel_size, dilation=dilation, padding=padding, stride=stride)
    mat = x_unfold.permute(0,2,1).contiguous().view(-1,x_unfold.shape[1])
    return mat

def forward_cache_activations(x, layer, key, prev_recur_proj_mat=None, act={"pre":OrderedDict(), "post":OrderedDict()} ):       
    if isinstance(layer, nn.Conv2d):
        if prev_recur_proj_mat is not None:
            act["pre"][key]=torch.matmul(reshape_conv_input_activation(deepcopy(x.clone().detach()), layer), prev_recur_proj_mat["pre"][key]).cpu().numpy()
            # Easier to project weights and then convolve.
            weight =torch.mm(layer.weight.data.flatten(1), prev_recur_proj_mat["pre"][key].transpose(0,1)).view_as(layer.weight.data)
            bias = None if layer.bias is None else layer.bias.data
            stride = layer.stride
            padding =  layer.padding 
            dilation = layer.dilation
            x = F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=1)
            act["post"][key]=deepcopy(x.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, x.shape[1]))
            x = torch.matmul( x.permute(0,2,3,1).reshape(-1, x.shape[1]).contiguous(), prev_recur_proj_mat["post"][key] ).reshape(x.shape[0],x.shape[2], x.shape[3], x.shape[1]).permute(0,3,1,2).contiguous()
        else:
            act["pre"][key]=reshape_conv_input_activation(deepcopy(x.clone().detach()), layer).cpu().numpy()
            x = layer(x)
            act["post"][key]=deepcopy(x.permute(0,2,3,1).clone().detach().cpu().numpy().reshape(-1, x.shape[1]))                            
    elif isinstance(layer, nn.Linear):
        if prev_recur_proj_mat is not None:
            act["pre"][key]=torch.matmul(deepcopy(x.clone().detach()), prev_recur_proj_mat["pre"][key]).cpu().numpy()
            x = torch.matmul( x, prev_recur_proj_mat["pre"][key] )     
        act["pre"][key]=deepcopy(x.clone().detach().cpu().numpy())
        x = layer(x)
        act["post"][key]= deepcopy(x.clone().detach().cpu().numpy())  
        if prev_recur_proj_mat is not None:
             x = torch.matmul( x, prev_recur_proj_mat["post"][key] )            
    else:
        x = layer(x)
    return act, x


def auto_get_activations(x, layers, block_key, prev_recur_proj_mat, act):
    if isinstance(layers, nn.Sequential):
        layer_ind = 0
        for layer in layers:
            layer_key = f"{block_key}.layer{layer_ind}"
            act, x = get_activations_layer(x, layer, layer_key, prev_recur_proj_mat, act)
            layer_ind+=1 
        return act, x
    else:
        return get_activations_layer(x, layers, block_key, prev_recur_proj_mat, act)

def get_activations_layer(x, layer, layer_key, prev_recur_proj_mat, act):
    if not (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or hasattr(layer, "get_activations") ) :
        x = layer(x)
        return act, x
    if hasattr(layer, "get_activations"):
        return layer.get_activations(x, layer_key, prev_recur_proj_mat, act)  
    else:
        return forward_cache_activations(x, layer, layer_key, prev_recur_proj_mat, act) 

def auto_project_weights(layers, block_key, projection_mat_dict, proj_classifier = True):
    if isinstance(layers, nn.Sequential):
        layer_ind = 0
        for layer_number, layer in enumerate(layers):
            layer_key = f"{block_key}.layer{layer_ind}"
            if layer_number == len(layers)-1 :
                project_weights_layer(layer, layer_key, projection_mat_dict, proj_classifier)
            else:
                project_weights_layer(layer, layer_key, projection_mat_dict, True)
            layer_ind+=1 
    else:
        project_weights_layer(layers, block_key, projection_mat_dict, proj_classifier)
    return 

def project_weights_layer(layer, layer_key, projection_mat_dict, post_projection = True):
    if not (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) or hasattr(layer, "project_weights") ) :
        return
    if hasattr(layer, "project_weights"):
        layer.project_weights(layer_key, projection_mat_dict)  
    else:
        if post_projection:
            # Check if layer key exists in projection matrix dictionary
            if f"{layer_key}" not in projection_mat_dict["post"] or f"{layer_key}" not in projection_mat_dict["pre"]:
                return
            
            layer.weight.data = torch.mm(projection_mat_dict["post"][f"{layer_key}"].transpose(0,1) ,torch.mm(layer.weight.data.flatten(1), projection_mat_dict["pre"][f"{layer_key}"].transpose(0,1))).view_as(layer.weight.data)
            if layer.bias is not None:
                layer.bias.data = torch.mm( layer.bias.data.unsqueeze(0), projection_mat_dict["post"][f"{layer_key}"]).squeeze(0)
        else:
            if f"{layer_key}" not in projection_mat_dict["pre"]:
                return
            layer.weight.data = torch.mm(layer.weight.data.flatten(1), projection_mat_dict["pre"][f"{layer_key}"].transpose(0,1)).view_as(layer.weight.data)

def attach_sap_methods(model):
    from types import MethodType
    
    def get_activations_resnet_cifar(self, x, prev_recur_proj_mat=None, act=None):
        if act is None: act={"pre":OrderedDict(), "post":OrderedDict()}
        # Assuming structure similar to ResNet_cifar
        if hasattr(self, 'conv1'):
            act, out = auto_get_activations(x, self.conv1, "conv1", prev_recur_proj_mat, act)
            out = F.relu(self.bn1(out))
        else:
            # Fallback or other structure
            pass
            
        if hasattr(self, 'layer1'): act, out = auto_get_activations(out, self.layer1, "layer1", prev_recur_proj_mat, act)
        if hasattr(self, 'layer2'): act, out = auto_get_activations(out, self.layer2, "layer2", prev_recur_proj_mat, act)
        if hasattr(self, 'layer3'): act, out = auto_get_activations(out, self.layer3, "layer3", prev_recur_proj_mat, act)
        if hasattr(self, 'layer4'): act, out = auto_get_activations(out, self.layer4, "layer4", prev_recur_proj_mat, act)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        
        if hasattr(self, 'fc'):
            act, out = auto_get_activations(out, self.fc, f"fc", prev_recur_proj_mat, act)
        elif hasattr(self, 'linear'):
            act, out = auto_get_activations(out, self.linear, f"linear", prev_recur_proj_mat, act)
            
        return act

    def get_activations_resnet_imagenet(self, x, prev_recur_proj_mat=None, act=None):
        if act is None: act={"pre":OrderedDict(), "post":OrderedDict()}
        # Assuming structure similar to ResNet_imagenet
        act, out = auto_get_activations(x, self.conv1, "conv1", prev_recur_proj_mat, act)
        # Check for maxpool existence
        if hasattr(self, 'maxpool'):
            out = self.maxpool(F.relu(self.bn1(out)))
        else:
            out = F.relu(self.bn1(out))
            
        act, out = auto_get_activations(out, self.layer1, "layer1", prev_recur_proj_mat, act)
        act, out = auto_get_activations(out, self.layer2, "layer2", prev_recur_proj_mat, act)
        act, out = auto_get_activations(out, self.layer3, "layer3", prev_recur_proj_mat, act)
        act, out = auto_get_activations(out, self.layer4, "layer4", prev_recur_proj_mat, act)
        
        if hasattr(self, 'avgpool'):
             out = self.avgpool(out)
        else:
             out = F.avg_pool2d(out, 4) # Fallback
             
        out = out.view(out.size(0), -1)
        act, out = auto_get_activations(out, self.fc, f"fc", prev_recur_proj_mat, act)
        return act

    def project_weights_resnet(self, projection_mat_dict, proj_classifier=False):
        auto_project_weights(self.conv1, f"conv1", projection_mat_dict) 
        auto_project_weights(self.layer1, f"layer1", projection_mat_dict) 
        auto_project_weights(self.layer2, f"layer2", projection_mat_dict) 
        auto_project_weights(self.layer3, f"layer3", projection_mat_dict) 
        auto_project_weights(self.layer4, f"layer4", projection_mat_dict) 
        
        if hasattr(self, 'fc'):
            auto_project_weights(self.fc, f"fc", projection_mat_dict, proj_classifier=proj_classifier)
        elif hasattr(self, 'linear'):
            auto_project_weights(self.linear, f"linear", projection_mat_dict, proj_classifier=proj_classifier)
        
    # Check attributes to decide
    if hasattr(model, 'layer1') and hasattr(model, 'conv1'):
        if not hasattr(model, 'get_activations'):
            # Simple heuristic: if maxpool exists, likely imagenet version
            if hasattr(model, 'maxpool'):
                 model.get_activations = MethodType(get_activations_resnet_imagenet, model)
            else:
                 model.get_activations = MethodType(get_activations_resnet_cifar, model)
        
        if not hasattr(model, 'project_weights'):
            model.project_weights = MethodType(project_weights_resnet, model)
            
    return model
