import torch
from torch import nn
from torchvision.utils import make_grid
from base import BaseTrainer, MetricTracker, ConfusionTracker
from utils import inf_loop, tb_projector_resize, plot_classes_preds, close_all_plots
from utils import register_forward_hook_layer
import numpy as np
from copy import deepcopy

import data_loader.data_augmentation as module_DA
import data_loader.data_sampling as module_sampling


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, curve_metric_ftns, optimizer, config, classes, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, curve_metric_ftns, optimizer, config, classes, device)
        self.config = config
        self.device = device
        
        if len_epoch is None:
            self.data_loader = data_loader
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
            
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_name = config['lr_scheduler']['type'] if 'lr_scheduler' in config.config.keys() else None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

        self.confusion_key = 'confusion'
        self.train_confusion = ConfusionTracker(*[self.confusion_key], writer=self.writer, classes=self.classes)
        self.valid_confusion = ConfusionTracker(*[self.confusion_key], writer=self.writer, classes=self.classes)

        self.softmax = nn.Softmax(dim=0)
        self.preds_item_cnt = 5
        self.prediction_images, self.prediction_labels = None, None
        self.prediction_preds, self.prediction_probs = None, None

        # Hook for DA
        if self.cfg_da is not None:            
            if self.DA == 'cutmix': self.DA_ftns = module_DA.CutMix(writer=self.writer, **self.DAargs)
            
            elif self.DA == 'mixup': self.DA_ftns = module_DA.MixUp(writer=self.writer, **self.DAargs)
            
            elif self.DA == 'classical_augmentation': self.DA_ftns = module_DA.Classical_Augmentation(writer=self.writer, **self.DAargs)
            
        # Clear the gradients of all optimized variables 
        self.optimizer.zero_grad()

    def _curve_metrics(self, mode='training'):
        for met in self.curve_metric_ftns:
            if mode=='training':
                actual_vector = self.train_confusion.get_actual_vector(self.confusion_key)
                probability_vector = self.train_confusion.get_probability_vector(self.confusion_key)
            else:
                actual_vector = self.valid_confusion.get_actual_vector(self.confusion_key)
                probability_vector = self.valid_confusion.get_probability_vector(self.confusion_key)
            curve_fig = met(actual_vector, probability_vector, self.classes)
            self.writer.add_figure(met.__name__, curve_fig)
            if self.save_performance_plot: curve_fig.savefig(self.output_dir / f'{met.__name__}_{mode}.png', bbox_inches='tight')
            
    def _get_a_log(self, epoch):        
        log = self.train_metrics.result()        
        log_confusion = self.train_confusion.result()

        if self.do_validation:
            val_log, val_confusion = self._valid_epoch(epoch) # Validation Result
            log.update(**{'val_'+k : v for k, v in val_log.items()})
            log_confusion.update(**{'val_'+k : v for k, v in val_confusion.items()})
        log.update(log_confusion)           
        return log
    
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch. Start with 1.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.train_confusion.reset()
        label_img, features, class_labels = None, None, []
        data_channel = None
        
        # Hook
        if self.cfg_da is not None:
            hook = register_forward_hook_layer(self.model, self.DA_ftns.forward_pre_hook if self.pre_hook else self.DA_ftns.forward_hook, **self.hookargs)
        
        for batch_idx, (data, target) in enumerate(self.data_loader):
            batch_num = (epoch - 1) * self.len_epoch + batch_idx
            self.writer.set_step(batch_num, 'batch_train')
                
            # 1. To move Torch to the GPU or CPU
            if self.sampling is not None: data, target = self._sampling(data, target)
            data, target = data.to(self.device), target.to(self.device)

            # Compute prediction error
            # 2. Forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(data)
            logit, predict = torch.max(output, 1)
            loss = self._loss(output, target, logit)
            if self.cfg_da is not None: loss = self._da_loss(output, target, logit, loss)
                
            # 3. Backward pass: compute gradient of the loss with respect to model parameters
            if self.accumulation_steps is not None: loss = loss / self.accumulation_steps
            loss.backward()
            # 4-1. Perform a single optimization step (parameter update)
            # 4-2. Clear the gradients of all optimized variables 
            if self.accumulation_steps is not None:
                if batch_idx == self.len_epoch-1 or self.len_epoch == 1 or (batch_idx+1) % self.accumulation_steps == 0: 
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else: 
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            # 5. Update the Result
            # 5-1. loss
            self.train_metrics.update('loss', loss.item())
            
            # 5-2. confusion matrix 
            confusion_content = {'actual':target.cpu().tolist(), 'predict':predict.cpu().tolist()}
            if self.curve_metric_ftns is not None: confusion_content['probability']=[self.softmax(el).tolist() for el in output.detach().cpu()]
            self.train_confusion.update(self.confusion_key, confusion_content, img_update=False)
            
            confusion_obj = self.train_confusion.get_confusion_obj(self.confusion_key)
            for met in self.metric_ftns:# pycm version
                met_name_idx = self.metrics_class_index[met.__name__]
                use_confusion_obj = deepcopy(confusion_obj)
                if met_name_idx is None: self.train_metrics.update(met.__name__, met(use_confusion_obj, self.classes))
                else: self.train_metrics.update(met.__name__, met(use_confusion_obj, self.classes, met_name_idx))
            
            # 5-3-1. Projector
            # The data concerning the projector is collected with each batch and will be updated after all batches are completed.
            # See 5-3-2.
            if self.train_projector and epoch == 1:                
                label_img, features = tb_projector_resize(data.detach().cpu().clone(), label_img, features)
                class_labels.extend([str(self.classes[lab]) for lab in target.cpu().tolist()])
                
            # 6. Print the result
            if batch_idx % self.log_step == 0 or batch_idx == self.len_epoch-1 or self.len_epoch == 1:
                self.logger.debug(f'Train Epoch: {epoch} {self._progress(batch_idx)} | Acc: {confusion_obj.Overall_ACC:.6f} | Loss: {loss.item():.6f}')
                self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
            
            if (batch_idx == self.len_epoch-2 or self.len_epoch == 1) and self.tensorboard_pred_plot: 
                # last batch -1 > To minimize batches with a length of 1 as much as possible.
                # If you want to modify the last batch, pretend that self.len_epoch-2 is self.len_epoch-1.
                self.prediction_images, self.prediction_labels = data.cpu()[-self.preds_item_cnt:], [self.classes[lab] for lab in target.cpu().tolist()[-self.preds_item_cnt:]]
                data_channel = self.prediction_images.shape[1]
                preds = np.squeeze(predict[-self.preds_item_cnt:].detach().cpu().numpy())
                preds = preds if len(target)!=1 else np.array([preds]) # For batches with length of 1                 
                use_prob = self.train_confusion.get_probability_vector(self.confusion_key)[-len(preds):] if self.curve_metric_ftns is not None \
                           else [self.softmax(el).tolist() for el in output[-len(preds):].detach().cpu()]
                self.prediction_preds = [self.classes[lab] for lab in preds]
                self.prediction_probs = [el[i] for i, el in zip(preds, use_prob)]          
            
            if batch_idx == self.len_epoch: break
        
        if self.cfg_da is not None: hook.remove()
        # 5-3-2. Update the curve plot and projector
        self.writer.set_step(epoch-1)
        if self.curve_metric_ftns is not None: self._curve_metrics(mode='training')
        if self.train_projector and epoch == 1: self.writer.add_embedding('DataEmbedding', features, metadata=class_labels, label_img=label_img)
        # 5-3-3. Upate the example of predtion
        if self.tensorboard_pred_plot:
            self.writer.add_figure('Prediction', plot_classes_preds(self.prediction_images, self.prediction_labels, 
                                                                    self.prediction_preds, self.prediction_probs,
                                                                    one_channel = True if data_channel == 1 else False, return_plot=True))
        close_all_plots()
        self.prediction_images, self.prediction_labels = None, None
        self.prediction_preds, self.prediction_probs = None, None   
        
        # 6. Upate the lr scheduler
        if self.lr_scheduler is not None:
            self.writer.set_step(epoch-1)
            self.writer.add_scalar('lr_schedule', self.optimizer.param_groups[0]['lr'])
            if self.lr_scheduler_name == 'ReduceLROnPlateau': self.lr_scheduler.step(val_log['loss'])
            else: self.lr_scheduler.step()
        
        # 7. setting result     
        return self._get_a_log(epoch)
    
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        self.valid_confusion.reset()
        label_img, features, class_labels = None, None, []
        data_channel = None
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                batch_num = (epoch - 1) * len(self.valid_data_loader) + batch_idx
                self.writer.set_step(batch_num, 'batch_valid')
                
                # 1. To move Torch to the GPU or CPU
                data, target = data.to(self.device), target.to(self.device)

                # Compute prediction error
                # 2. Forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                logit, predict = torch.max(output, 1) 
                loss = self._loss(output, target, logit)

                # 3. Update the loss
                self.valid_metrics.update('loss', loss.item())
                # 4. Update the confusion matrix and input data
                confusion_content = {'actual':target.cpu().tolist(), 'predict':predict.cpu().tolist()}
                if self.curve_metric_ftns is not None: confusion_content['probability']=[self.softmax(el).tolist() for el in output.detach().cpu()]
                self.valid_confusion.update(self.confusion_key, confusion_content, img_update=False)
                    
                confusion_obj = self.valid_confusion.get_confusion_obj(self.confusion_key)
                for met in self.metric_ftns:# pycm version
                    met_name_idx = self.metrics_class_index[met.__name__]
                    use_confusion_obj = deepcopy(confusion_obj)
                    if met_name_idx is None: self.valid_metrics.update(met.__name__, met(use_confusion_obj, self.classes))
                    else: self.valid_metrics.update(met.__name__, met(use_confusion_obj, self.classes, met_name_idx))
                
                    
                if batch_idx % self.log_step == 0: self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
                
                # 4-1. Update the Projector
                if self.valid_projector and epoch == 1:                    
                    label_img, features = tb_projector_resize(data.detach().cpu().clone(), label_img, features)
                    class_labels.extend([str(self.classes[lab]) for lab in target.cpu().tolist()])
                
                if (batch_idx == len(self.valid_data_loader)-2 or len(self.valid_data_loader) == 1) and self.tensorboard_pred_plot:
                    # last batch -1 > To minimize batches with a length of 1 as much as possible.
                    # If you want to modify the last batch, pretend that self.len_epoch-2 is self.len_epoch-1.
                    self.prediction_images, self.prediction_labels = data.cpu()[-self.preds_item_cnt:], [self.classes[lab] for lab in target.cpu().tolist()[-self.preds_item_cnt:]]
                    data_channel = self.prediction_images.shape[1]
                    preds = np.squeeze(predict[-self.preds_item_cnt:].detach().cpu().numpy())
                    preds = preds if len(target)!=1 else np.array([preds]) # For batches with length of 1  
                    use_prob = self.valid_confusion.get_probability_vector(self.confusion_key)[-len(preds):] if self.curve_metric_ftns is not None \
                               else [self.softmax(el).tolist() for el in output[-len(preds):].detach().cpu()]
                    self.prediction_preds = [self.classes[lab] for lab in preds]
                    self.prediction_probs = [el[i] for i, el in zip(preds, use_prob)]  
                    
        # 4-2. Update the curve plot and projector
        self.writer.set_step(epoch-1, 'valid')
        if self.curve_metric_ftns is not None: self._curve_metrics(mode='validation')
        if self.valid_projector and epoch == 1: self.writer.add_embedding('DataEmbedding', features, metadata=class_labels, label_img=label_img)
        # 4-3. Upate the example of predtion
        if self.tensorboard_pred_plot:
            self.writer.add_figure('Prediction', plot_classes_preds(self.prediction_images, self.prediction_labels, 
                                                                    self.prediction_preds, self.prediction_probs,
                                                                    one_channel = True if data_channel == 1 else False, return_plot=True))
        close_all_plots()
        self.prediction_images, self.prediction_labels = None, None
        self.prediction_preds, self.prediction_probs = None, None   
        
        # 5. Print the result
        confusion_obj = self.valid_confusion.get_confusion_obj(self.confusion_key)
        self.logger.debug(f'Valid Epoch: {epoch} | Acc: {confusion_obj.Overall_ACC:.6f} | Loss: {loss.item():.6f}')
        
        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(), self.valid_confusion.result()

    def _loss(self, output, target, logit):
        if self.loss_fn_name != 'bce_loss': loss = self.criterion(output, target)
        else: loss =  self.criterion(logit, target.type(torch.DoubleTensor).to(self.device))
        return loss
        
    def _da_loss(self, output, target, logit, loss):
        if self.DA == 'cutmix':
            if self.DA_ftns.rand_index() is None: return loss
            random_index = self.DA_ftns.rand_index()
            if len(random_index) != len(target): raise ValueError('Target and the number of shuffled indexes do not match.')
            random_loss = self._loss(output, target[random_index], logit).item()
            lam = self.DA_ftns.lam()
            loss = loss*lam +  random_loss*(1.-lam)
            
        elif self.DA == 'mixup':
            if self.DA_ftns.rand_index() is None: return loss
            random_index = self.DA_ftns.rand_index()
            if len(random_index) != len(target): raise ValueError('Target and the number of shuffled indexes do not match.')
            random_loss = self._loss(output, target[random_index], logit).item()
            lam = self.DA_ftns.lam()
            loss = loss*lam +  random_loss*(1.-lam)
            
        self.DA_ftns.reset()
        
        return loss
    
    def _progress(self, batch_idx):
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        
        str_diff = len(str(total))-len(str(current))
        current_str = str(current) if str_diff == 0 else ' '*str_diff+str(current)
        percentage = f'{100.0 * (current/total):.0f}' 
        return f'[{current_str}/{total} ({percentage:2s})%]'
    
    def _sampling(self, data, target):
        if 'down' in self.sampling_type:
            if 'random' in self.sampling_name: return module_sampling.random_downsampling(data, target)
        elif 'up' in self.sampling_type:
            # data, target = self._upsampling(data, target)
            pass
        else: TypeError('The applicable types are up or down.')
        