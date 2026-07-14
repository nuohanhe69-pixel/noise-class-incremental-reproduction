from __future__ import print_function
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F

def get_representation_matrix_mislabeled(net, device, data_loader, sample_indexs=[], num_classes=10, prev_recur_proj_mat=None,
                               samples_per_set=1000, max_batch_size=150, max_samples=50000, set_name="Clean Set"):
    """
    Extract activation matrix from neural network layers
    """
    if sample_indexs:
        # Sort data by class and collect samples for required classes
        dataset = data_loader.dataset
        samples_list = []
        r = np.arange(len(sample_indexs))
        np.random.shuffle(r)
        sample_indexs = np.array(sample_indexs)[r[:samples_per_set]].tolist()
        for index in sample_indexs:
            samples_list.append(dataset[index][0])
        sample_tensor = torch.stack(samples_list, 0).to(device)
        if len(sample_tensor.shape) < 4:
            sample_tensor = sample_tensor.unsqueeze(1)
            # print(sample_tensor.shape)
        # Get activations as dictionary
        activations = None
        net.eval()
        for batch in tqdm(torch.split(sample_tensor, max_batch_size, dim=0), desc=f"Extracting representation for {set_name}"):
            try:
                batch_activations = net.get_activations(batch, prev_recur_proj_mat)
            except:
                batch_activations = net.module.get_activations(batch, prev_recur_proj_mat)
            
            # Compress batch immediately
            for loc in batch_activations.keys():
                for key in batch_activations[loc].keys():
                    if batch_activations[loc][key].shape[0] > (int(max_samples / (sample_tensor.shape[0] / max_batch_size)) + 1):
                        # Shuffle and return subset
                        r = np.arange(batch_activations[loc][key].shape[0])
                        np.random.shuffle(r)
                        b = r[:(int(max_samples / (sample_tensor.shape[0] / max_batch_size)) + 1)]
                        batch_activations[loc][key] = batch_activations[loc][key][b].copy()
            
            # Concatenate samples
            if activations:
                for loc in batch_activations.keys():
                    for key in batch_activations[loc].keys():
                        activations[loc][key] = np.concatenate([activations[loc][key], batch_activations[loc][key]], 0)
            else:
                activations = batch_activations
        
        # Final check to reduce sample size
        sampled_activations = {"pre": OrderedDict(), "post": OrderedDict()}
        for loc in batch_activations.keys():
            for key in batch_activations[loc].keys():
                if activations[loc][key].shape[0] > max_samples:
                    # Shuffle and return subset
                    r = np.arange(activations[loc][key].shape[0])
                    np.random.shuffle(r)
                    b = r[:max_samples]
                    sampled_activations[loc][key] = activations[loc][key][b].copy()
                else:
                    sampled_activations[loc][key] = activations[loc][key].copy()
        
        # Transpose activations
        loc_keys = list(sampled_activations.keys())
        act_keys = list(sampled_activations[loc_keys[0]].keys())
        mat_dict = {loc: OrderedDict() for loc in loc_keys}
        for loc in loc_keys:
            for act in list(sampled_activations[loc].keys()):
                activation = sampled_activations[loc][act].transpose()
                mat_dict[loc][act] = activation
        
        # Print representation shape
        # for loc in loc_keys:
        #     print('-' * 30)
        #     print(f'Representation Matrix {loc} Layer for {set_name}')
        #     print('-' * 30)
        #     for act in list(sampled_activations[loc].keys()):
        #         print(f' Layer {act} : [{mat_dict[loc][act].shape}]')
        #     print('-' * 30)
        return mat_dict
    else:
        return {loc: OrderedDict() for loc in ["pre", "post"]}


def get_SVD(mat_dict, device, set_name="SVD"):
    """
    Perform SVD decomposition on activation matrices to get basis vectors and singular values
    """
    feature_dict = {"pre": OrderedDict(), "post": OrderedDict()}
    s_dict = {"pre": OrderedDict(), "post": OrderedDict()}
    
    for loc in mat_dict.keys():
        for act in tqdm(mat_dict[loc].keys(), desc=f"{loc}layer - SVD for {set_name}"):
            activation = torch.Tensor(mat_dict[loc][act]).to(device)
            U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
            U = U.cpu().numpy()
            S = S.cpu().numpy()
            feature_dict[loc][act] = U
            s_dict[loc][act] = S
    
    return feature_dict, s_dict


def get_projections(feature_mat_retain_dict, feature_mat_unlearn_dict, projection_type, device):
    """
    Construct projection matrices by combining basis vectors of clean and contaminated spaces
    """
    feature_mat = {"pre": OrderedDict(), "post": OrderedDict()}
    for loc in feature_mat_retain_dict.keys():
        for act in feature_mat_retain_dict[loc].keys():
            Ur = feature_mat_retain_dict[loc][act]
            Uf = feature_mat_unlearn_dict[loc][act]
            Mr = torch.mm(Ur, Ur.transpose(0, 1))
            Mf = torch.mm(Uf, Uf.transpose(0, 1))
            I = torch.eye(Mf.shape[0]).to(device)
            Mri = torch.mm(Mr, Mf)  # Intersection in terms of retain space basis
            Mfi = torch.mm(Mf, Mr)  # Intersection in terms of forget space basis
            # Select type of projection.
            if projection_type == "baseline":
                feature_mat[loc][act] = I
            elif projection_type == "Mr":
                feature_mat[loc][act] = Mr
            elif projection_type == "I-Mf":
                feature_mat[loc][act] = I - Mf
            elif projection_type == "Mr-Mi":
                feature_mat[loc][act] = Mr - torch.mm(Mr, Mf)
            elif projection_type == "I-(Mf-Mi)":
                feature_mat[loc][act] = I - (Mf - torch.mm(Mf, Mr))
            else:
                raise ValueError
    return feature_mat



def activation_projection_based_unlearning(args, model, 
                                           train_loaders, 
                                           val_loaders, 
                                           test_loader, 
                                           trainset,
                                           device, 
                                           round=0, 
                                           prev_recur_proj_mat=None):
    """
    Core implementation function of SAP algorithm
    """
    train_loader, train_loader_clean, train_loader_corrupt = train_loaders
    val_loader, val_loader_clean, val_loader_corrupt = val_loaders
    
    # Initialize model
    if round == 0:
        inference_model = copy.deepcopy(model)
    else:
        inference_model = copy.deepcopy(model)
        try:
            inference_model.module.project_weights(prev_recur_proj_mat, args.project_classifier)
        except:
            inference_model.project_weights(prev_recur_proj_mat, args.project_classifier)
    
    # Initial evaluation
    if train_loader_corrupt:
        corrupt_acc, clean_loss = test(inference_model, device, train_loader_corrupt, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Corrupt Train Set", verbose=prev_recur_proj_mat is None)
    
    if args.use_valset:
        full_acc, full_loss = test(inference_model, device, val_loader, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Full Val Set", verbose=prev_recur_proj_mat is None)
        base_metric = full_acc  
        if val_loader_clean:
            clean_acc, clean_loss = test(inference_model, device, val_loader_clean, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Clean Val Set", verbose=prev_recur_proj_mat is None)
            base_metric = clean_acc
    else:
        full_acc, full_loss = test(inference_model, device, train_loader, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Full Train Set", verbose=prev_recur_proj_mat is None)
        base_metric = full_acc
        if train_loader_clean:
            clean_acc, clean_loss = test(inference_model, device, train_loader_clean, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Clean Train Set", verbose=prev_recur_proj_mat is None)
            base_metric = clean_acc
    
    test_acc, test_loss = test(inference_model, device, test_loader, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Test Set", verbose=prev_recur_proj_mat is None)
    
    # Get basis vectors for clean set
    mat_retain_dict = get_representation_matrix_mislabeled(model, device, train_loader_clean, sample_indexs=np.arange(len(train_loader_clean.dataset)).tolist(), num_classes=args.num_classes,
                                                         prev_recur_proj_mat=prev_recur_proj_mat, samples_per_set=args.retain_samples,
                                                         max_batch_size=args.max_batch_size, max_samples=args.max_samples, set_name="Clean Set")
    full_feature_retain_dict, full_s_retain_dict = get_SVD(mat_retain_dict, device, f"SVD Clean Set")
    
    # Get basis vectors for corrupt set
    mat_unlearn_dict = get_representation_matrix_mislabeled(model, device, train_loader_corrupt, sample_indexs=np.arange(len(train_loader_corrupt.dataset)).tolist(), num_classes=args.num_classes,
                                                         prev_recur_proj_mat=prev_recur_proj_mat, samples_per_set=args.forget_samples,
                                                         max_batch_size=args.max_batch_size, max_samples=args.max_samples, set_name="Corrupt Set")
    full_feature_unlearn_dict, full_s_unlearn_dict = get_SVD(mat_unlearn_dict, device, f"SVD Corrupt Set")
    
    best_metric = base_metric
    next_recur_proj_mat = None
    unlearnt_model = copy.deepcopy(model)
    num_layer = len(full_s_retain_dict["pre"])
    
    for mode in args.mode:
        # Iterate over all retain modes
        for mode_forget in args.mode_forget:
            # Iterate over all forget modes
            for projection_type in args.projection_type:
                # Iterate over all projection types
                for start in args.start_layer:
                    for end in args.end_layer:
                        # Set eps_threshold and scale_coff_list for retain set and get basis
                        if mode == "baseline":
                            scale_coff_list = [0]
                            eps_threshold = None
                        elif mode == "gpm":
                            scale_coff_list = [0]
                            eps_threshold = args.gpm_eps
                        else:
                            scale_coff_list = args.scale_coff
                            eps_threshold = None
                        
                        for projection_location in args.projection_location:
                            # Iterate over all projection locations
                            for alpha in scale_coff_list:
                                # Set eps_threshold and scale_coff_list for forget set and get basis
                                if mode_forget is None:
                                    mode_forget = mode
                                    scale_coff_list_forget = [alpha]
                                    eps_threshold_forget = eps_threshold
                                elif mode_forget == "baseline":
                                    scale_coff_list_forget = [0]
                                    eps_threshold_forget = None
                                elif mode_forget == "gpm":
                                    scale_coff_list_forget = [0]
                                    eps_threshold_forget = args.gpm_eps
                                else:
                                    scale_coff_list_forget = args.scale_coff_forget
                                    eps_threshold_forget = None
                                
                                # Get feature matrix for retain space Mr
                                feature_retain_dict = select_basis(full_feature_retain_dict, full_s_retain_dict, eps_threshold)
                                feature_mat_retain_dict = get_scaled_feature_mat(feature_retain_dict, full_s_retain_dict, mode, alpha, device)
                                
                                for alpha_forget in scale_coff_list_forget:
                                    # Get feature matrix for forget space Mf
                                    feature_unlearn_dict = select_basis(full_feature_unlearn_dict, full_s_unlearn_dict, eps_threshold_forget)
                                    feature_mat_unlearn_dict = get_scaled_feature_mat(feature_unlearn_dict, full_s_unlearn_dict, mode_forget, alpha_forget, device)
                                    
                                    # Get projection matrix using Mf and Mr
                                    projection_mat = get_projections(feature_mat_retain_dict, feature_mat_unlearn_dict, projection_type, device)
                                    
                                    # Modify projection matrix to respect considered layers and projection locations
                                    modified_projection_mat = {"pre": OrderedDict(), "post": OrderedDict()}
                                    for loc in projection_mat.keys():
                                        for i, act in enumerate(projection_mat[loc].keys()):
                                            if i < start:
                                                modified_projection_mat[loc][act] = torch.eye(projection_mat[loc][act].shape[0]).to(device)
                                            elif i > num_layer - end:
                                                modified_projection_mat[loc][act] = torch.eye(projection_mat[loc][act].shape[0]).to(device)
                                            elif projection_location != "all" and (loc != projection_location):
                                                modified_projection_mat[loc][act] = torch.eye(projection_mat[loc][act].shape[0]).to(device)
                                            else:
                                                modified_projection_mat[loc][act] = projection_mat[loc][act]
                                            if prev_recur_proj_mat is not None:
                                                modified_projection_mat[loc][act] = torch.matmul(prev_recur_proj_mat[loc][act], modified_projection_mat[loc][act])
                                    
                                    # Copy original training model and project its weights
                                    inference_model = copy.deepcopy(model)
                                    try:
                                        inference_model.module.project_weights(modified_projection_mat, args.project_classifier)
                                    except:
                                        inference_model.project_weights(modified_projection_mat, args.project_classifier)
                                    
                                    # Evaluate projection
                                    if args.use_valset:
                                        if val_loader_clean:
                                            clean_acc, clean_loss = test(inference_model, device, val_loader_clean, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Clean Val Set", verbose=False)
                                        full_acc, full_loss = test(inference_model, device, val_loader, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Full Val Set", verbose=False)
                                    else:
                                        if train_loader_clean:
                                            clean_acc, clean_loss = test(inference_model, device, train_loader_clean, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Clean Train Set", verbose=False)
                                        full_acc, full_loss = test(inference_model, device, train_loader, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Full Train Set", verbose=False)
                                    
                                    test_acc, test_loss = test(inference_model, device, test_loader, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Test Set", verbose=False)
                                    
                                    if train_loader_corrupt:
                                        corrupt_acc, clean_loss = test(inference_model, device, train_loader_corrupt, class_label_names=args.class_label_names, num_classes=args.num_classes, set_name="Corrupt Train Set", verbose=False)
                                    
                                    # Determine best model
                                    if train_loader_clean or val_loader_clean:
                                        metric = clean_acc
                                    else:
                                        metric = full_acc
                                    
                                    if metric > best_metric:
                                        best_metric = metric
                                        unlearnt_model = inference_model
                                        next_recur_proj_mat = modified_projection_mat
                                        print(f"New best metric: {best_metric:.4f}")
    
    return unlearnt_model, next_recur_proj_mat


# Helper function: Test model
def test(model, device, data_loader, class_label_names=[], num_classes=10, set_name="Test Set", verbose=True):
    """
    Test model performance on given dataset
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in data_loader:
            data, target = batch[0], batch[1]
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(data_loader.dataset)
    
    if verbose:
        print('-' * 40)
        print(f'{set_name} Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({100. * correct / len(data_loader.dataset):.0f}%)')
        print('-' * 40)
    
    return 100. * correct / len(data_loader.dataset), test_loss


# Helper function: Select basis vectors
def select_basis(feature_dict, full_s_dict, threshold):
    """
    Select basis vectors based on cumulative singular value ratio
    """
    if threshold is None:
        return feature_dict
    out_feature_dict = {"pre": OrderedDict(), "post": OrderedDict()}
    for loc in feature_dict.keys():
        for act in feature_dict[loc].keys():
            U = feature_dict[loc][act]
            S = full_s_dict[loc][act]
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold) + 1
            out_feature_dict[loc][act] = U[:, :r]
    return out_feature_dict


# Helper function: Get scaled feature matrix
def get_scaled_feature_mat(feature_dict, full_s_dict, mode, alpha, device):
    """
    Scale basis vectors using importance coefficients
    """
    feature_mat_dict = {"pre": OrderedDict(), "post": OrderedDict()}
    for loc in feature_dict.keys():
        for act in feature_dict[loc].keys():
            U = torch.Tensor(feature_dict[loc][act]).to(device)
            S = full_s_dict[loc][act]
            if mode == "baseline":
                importance = torch.ones(U.shape[1]).to(device)
            elif mode == "gpm":
                importance = torch.ones(U.shape[1]).to(device)
            elif mode == "sgp":
                importance = torch.Tensor((alpha * S / ((alpha - 1) * S + max(S)))[:U.shape[1]]).to(device)
            elif mode == "sap":
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                importance = torch.Tensor((alpha * sval_ratio / ((alpha - 1) * sval_ratio + 1))[:U.shape[1]]).to(device)
            else:
                raise ValueError
            U.requires_grad = False
            feature_mat_dict[loc][act] = torch.mm(U, torch.diag(importance ** 0.5))
    return feature_mat_dict
