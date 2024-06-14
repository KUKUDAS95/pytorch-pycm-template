from base import BaseHook

import numpy as np
import torch
from utils import show_mix_result, close_all_plots

class MixUp(BaseHook):
    def __init__(self, beta:float=0.1, prob:float=0.5, writer=None):
        self.type = 'mixup'
        super().__init__(self.type, cols=['lam', 'rand_index'], writer=writer)
        self.beta, self.prob = beta, prob
        self.init_lam = np.random.beta(beta, beta)

    def lam(self):
        return self._data['lam'][self.type]
    
    def rand_index(self):
        return self._data['rand_index'][self.type]
    
    def forward_hook(self, module, input_data, output_data):
        r = np.random.rand(1)
        if self.beta > 0 and r < self.prob:
            device = output.get_device()
            output = self._run(output.detach().cpu().clone())
            if device != -1: output.cuda()
            return output
    
    def forward_pre_hook(self, module, input_data):
        r = np.random.rand(1)
        if self.beta > 0 and r < self.prob:
            
            use_data = input_data[0]
            device = use_data.get_device()
            
            use_data = self._run(use_data.detach().cpu().clone())
            if device != -1: use_data = use_data.cuda()
            return (use_data, )
    
    def _run(self, data):
        # Original code: https://github.com/facebookresearch/mixup-cifar10/blob/eaff31ab397a90fbc0a4aac71fb5311144b3608b/train.py#L141
        size =  data.size()  # B, C, H, W
        
        batch_size = size[0]
        
        rand_index = np.arange(0, batch_size)
        np.random.shuffle(rand_index)        
        self._data['rand_index'][self.type] = rand_index

        self._data['lam'][self.type] = self.init_lam
        
        mix_data = data.detach().clone()

        mix_data = self.init_lam * mix_data + (1 - self.init_lam) * mix_data[rand_index, :]
        
        if self.writer is not None:
            img_cnt = batch_size if batch_size < 5 else 5
            cut_data = []
            for idx in range(img_cnt):
                
                cut_data.append([(data[idx]), (mix_data[rand_index, :][idx]), (mix_data[idx])])  
            
            cut_data = torch.as_tensor(np.array(cut_data))
            self.writer.add_figure(f'input_{self.type}', show_mix_result(cut_data))

            close_all_plots()
        
        return mix_data
    
        
        
    
