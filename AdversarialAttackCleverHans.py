
"""
This is where the adversarial examples are generated. 
This script uses CleverHans 
"""

import torch
import numpy as np 
from torch.utils.data import DataLoader, TensorDataset
from cleverhans.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.attacks.projected_gradient_descent_hinge import projected_gradient_descent_hinge
from cleverhans.attacks.carlini_wagner_l2 import carlini_wagner_l2
# from Loss_Function import hinge_vec
# from autoattack import AutoAttack

# FGSM, from article "Explaining and Harnessing Adversarial Examples"
def FGSM(model, batch_size, test_loader, epsilon, norm):
    '''
        FGSM attack

                Parameters:
                        model: pytorch model under attack 
                        batch size (int) 
                        test_loader : data loader for images that will be perturbed
                        epsilon (double): parameter for the magnitude of the attack
                        norm: norm of the attack (np.inf or 2)
                Returns:
                        data_loader (PyTorch DataLoader): dataloader containing the adversarial images  
    ''' 
    X_adv = torch.empty([0], device="cuda")
    y = torch.empty(0, dtype=torch.int32, device="cuda")
    for images, labels in test_loader: 
        images, labels = images.to("cuda"), labels.to("cuda")
        x_ = fast_gradient_method(model.eval(), images, epsilon, norm, clip_min=0, clip_max=1, sanity_checks=False, targeted=False, y=labels) 
        X_adv = torch.cat((X_adv, x_), dim=0)
        y = torch.cat((y, labels.int()), dim=0)
    
    # make a dataloader for adversaries
    data_loader = make_dataloader(X_adv, y, batch_size)
    return data_loader


# PGD attack from article "Towards deep learning models resistant to adversarial attacks" 
def PGD(model, batch_size, test_loader, epsilon, eps_step, max_iter, norm, loss, early_stop=False): 
    '''
        PGD attack
                Parameters:
                        model: pytorch model under attack 
                        batch size (int) 
                        test_loader : data loader for images that will be perturbed
                        epsilon (double): parameter for the magnitude of the attack
                        eps_step (double): size of step taken by PGD every iteration
                        max iter (int): Number of PGD iterations performed
                        norm: norm of the attack (np.inf or 2)
                        loss : loss function used by PGD (cross-entropy or C&W)
                        early_stop (Boolean): True if PGD^es False for regular PGD

                Returns:
                        data_loader (PyTorch DataLoader): dataloader containing the adversarial images  
    ''' 
    X_adv = torch.empty([0], device="cuda")
    y = torch.empty(0, dtype=torch.int32, device="cuda")

    for images, labels in test_loader: 
        images, labels = images.to("cuda"), labels.to("cuda")
        
        # for untargeted attacks
        x_ = projected_gradient_descent(model, images, epsilon, eps_step, max_iter, norm, clip_min=0, clip_max=1, sanity_checks=False, y=labels, targeted=False, early_stop=early_stop, loss_fn=loss) 

        # update arrays
        X_adv = torch.cat((X_adv, x_), dim=0)
        y = torch.cat((y, labels.int()), dim=0)

    data_loader = data_loader = make_dataloader(X_adv, y, batch_size)
    return data_loader


# Carlini and Wagner, from article "Towards Evaluating the Robustness of Neural Networks"
def CW_L2(model, batch_size, test_loader, max_iter, confidence=0, learning_rate=5e-3, mode="SimCLR", codewords=[]):
    '''
        C&W_l2 attack
                Parameters:
                        model: pytorch model under attack 
                        batch size (int) 
                        test_loader : data loader for images that will be perturbed
                        max iter (int): Number of iterations performed
                        confidence: confidence param of the attack 
                        learning_rate: learning rate used by the attack

                Returns:
                        data_loader (PyTorch DataLoader): dataloader containing the adversarial images  
    ''' 
    
    X_adv = torch.empty([0], device="cuda")
    y = torch.empty(0, dtype=torch.int32, device="cuda")
    for images, labels in test_loader: 
        images, labels = images.to("cuda"), labels.to("cuda")
        x_ = carlini_wagner_l2(model.eval(), images, 10, clip_min=0, clip_max=1, targeted=False, y=labels, max_iterations=max_iter, confidence=confidence, lr=learning_rate, mode=mode, codewords=codewords)
        X_adv = torch.cat((X_adv, x_), dim=0)
        y = torch.cat((y, labels.int()), dim=0)
        
    data_loader = data_loader = make_dataloader(X_adv, y, batch_size)
    return data_loader


# PGD attack from article "Towards deep learning models resistant to adversarial attacks" 
# used for RegAdvt
def PGD_augmentation(model, images, epsilon, eps_step, max_iter, norm): 
    '''
        Augment batch images with PGD perturbations during training

                Parameters:
                        model: pytorch model under attack 
                        images : dimages to be augmented by adversarial training
                        epsilon (double): parameter for the magnitude of the attack
                        eps_step (double): size of step taken by PGD every iteration
                        max iter (int): Number of PGD iterations performed
                        norm: norm of the attack (np.inf or 2)
                Returns:
                        perturbed images  
    ''' 

    x_ = projected_gradient_descent(model.eval(), images, epsilon, eps_step, max_iter, norm, clip_min=0, clip_max=1, sanity_checks=False) 

    return x_


#helper function to make dataloaders
def make_dataloader(x_data, y_data, batch_size): 
    my_dataset = TensorDataset(x_data, y_data)
    data_loader = DataLoader(my_dataset, batch_size=batch_size)
    return data_loader
