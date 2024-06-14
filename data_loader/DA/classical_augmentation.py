from torchvision import transforms
from base import BaseHook
import numpy as np
import torch
from utils import show_mix_result, close_all_plots

augmentation = transforms.Compose([transforms.ColorJitter(brightness=(0.2,0.8), contrast=(0.2,0.8), saturation=(0.2,0.8), hue=(-0.5, 0.5)),])

class Classical_Augmentation(BaseHook):
    def __init__(self, beta:float=0.1, prob:float=0.5, writer=None):
        self.type = 'classical_augmentation'
        super().__init__(self.type, cols=['lam', 'rand_index'], writer=writer)
        self.beta, self.prob = beta, prob
        self.init_lam = np.random.beta(beta, beta)
    
    def forward_hook(self, module, input_data, output_data):
        r = np.random.rand(1)
        if r < self.prob:
            device = output.get_device()
            output = self._run(output.detach().cpu().clone())
            if device != -1: output.cuda()
            return output
    
    def forward_pre_hook(self, module, input_data):
        r = np.random.rand(1)
        if r < self.prob:
            
            use_data = input_data[0]
            device = use_data.get_device()
            
            use_data = self._run(use_data.detach().cpu().clone())
            if device != -1: use_data = use_data.cuda()
            return (use_data, )
    
    def _run(self, data):
        size =  data.size()  # B, C, H, W
        
        batch_size = size[0]
        
        aug_data = data.detach().clone()

        aug_data = augmentation(aug_data)
        
        if self.writer is not None:
            img_cnt = batch_size if batch_size < 5 else 5
            cut_data = []
            for idx in range(img_cnt):
                
                cut_data.append([(data[idx]), (aug_data[idx]), (aug_data[idx])])  
            
            cut_data = torch.as_tensor(np.array(cut_data))
            self.writer.add_figure(f'input_{self.type}', show_mix_result(cut_data))

            close_all_plots()
        
        return aug_data