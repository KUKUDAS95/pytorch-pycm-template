from pathlib import Path
import numpy as np
import pandas as pd
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
from copy import deepcopy

class ResultVisualization:
    def __init__(self, parent_dir, result_name,
                 test_dirname:str='test', test_filename:str='metrics', test_file_addtional_name:str='test',
                 positive_class_name=None):
        parent_dir = Path(parent_dir)
        self.name = result_name
        self.output_dir = parent_dir / 'output' / self.name
        
        self.test_dirname = test_dirname
        self.test_filename = test_filename
        self.test_file_addtional_name = test_file_addtional_name
        self.positive_class_name = positive_class_name
        self.sheet_list = ['training', 'validation', 'test']
        
        # output 중, metric.json 경로 정보 가져오기
        self.result_info = self._get_a_result_info()
        # self.df_dict = self._json2df()

    # output 중, metric.json 경로 정보 셋팅하는 함수
    def _read_json(self, json_path):
        json_content = None
        try:
            with open(json_path, 'r') as j:
                json_content = json.load(j)
        except Exception as inst: print(inst) 
        
        # 나중에 수정
        if 'auc' in json_content.keys(): 
            if self.positive_class_name is not None:
                json_content['auc']=json_content['auc'][self.positive_class_name]
                if 'val_auc' in json_content.keys(): json_content['val_auc']=json_content['val_auc'][self.positive_class_name]
            else: 
                print('Warning: The AUC scores from the training or validation dataset exist, but there are no positive classes set up to view the scores.')
                del json_content['auc']
                if 'val_auc' in json_content.keys(): del json_content['val_auc']
        return json_content
    
    def _get_a_category_info(self):
        cate_dict = {cate_path.name:cate_path for cate_path in self.output_dir.iterdir() if cate_path.is_dir()}
        cate_dict = dict(sorted(cate_dict.items()))
        return cate_dict
    
    def _get_a_cate_model_info(self):
        cate_dict = self._get_a_category_info()
        cate_model_dict = {k:None for k in cate_dict.keys()}
        for category, cate_path in cate_dict.items():
            _ = {model_path.name.lower():model_path for model_path in cate_path.iterdir() if model_path.is_dir()}
            cate_model_dict[category] = dict(sorted(_.items()))
        return cate_model_dict

    def _loss_sampling_da(self, json_path):
        json_content = self._read_json(json_path)
        loss = json_content['loss']
        da = json_content['data_augmentation']['type'] if 'data_augmentation' in json_content.keys() else None
        sampling =  json_content['data_sampling']['name'] if 'data_sampling' in json_content.keys() else None
        return loss, da, sampling
    
    def _get_a_result_info(self):
        cate_model_info = self._get_a_cate_model_info()
        metrics_dict = {k:{m:{} for m in cate_model_info[k].keys()} for k in cate_model_info.keys()}
        for category, cate_path in cate_model_info.items():
            for model, model_path in cate_path.items():
                # 카테고리, 모델마다 기법 적용에 따라 분류합니다.
                # 현재, 아무것도 적용하지 않은 None, 샘플링, 데이터 증강(DA), 2가지 이상 기법을 적용한 multi로 구분됩니다.
                tmp_dict = {metrics_path.name:metrics_path for metrics_path in model_path.iterdir() if metrics_path.is_dir()}
                metrics_None, metrics_sampling, metrics_DA, metrics_multi = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
                for metrics_cate, metrics_path in tmp_dict.items():
                    # config를 통해 어떤 기법이 적용되었는지 파악합니다.
                    config_path = sorted(Path(str(metrics_path).replace('output', 'models')).glob('config.json'))
                    if len(config_path) != 1: ValueError('The JSON file containing the training information could not be found.')
                    # 아직 템플릿은 DA, sampling만 지원하기 때문에, 가장 큰 구분자인 loss, DA, sampling 정보만 가져옵니다.
                    loss, da, sampling = self._loss_sampling_da(config_path[0])
                    if da is None and sampling is None: metrics_None[metrics_cate] = metrics_path
                    elif da is None and sampling is not None: metrics_sampling[metrics_cate] = metrics_path
                    elif da is not None and sampling is None: metrics_DA[metrics_cate] = metrics_path
                    else: metrics_multi[metrics_cate] = metrics_path
                metrics_None, metrics_sampling = dict(sorted(metrics_None.items())), dict(sorted(metrics_sampling.items()))
                metrics_DA, metrics_multi = dict(sorted(metrics_DA.items())), dict(sorted(metrics_multi.items()))
                metrics_None.update(metrics_sampling)
                metrics_None.update(metrics_DA)
                metrics_None.update(metrics_multi)
                for metrics_cate, metrics_path in metrics_None.items():
                    run_cate = '-'.join(metrics_cate.split('-')[:-1])
                    run_id = metrics_cate.split('-')[-1]
                    if run_cate not in metrics_dict[category][model].keys(): metrics_dict[category][model][run_cate] = {}
                    
                    # 훈련의 output을 가져온다.  
                    train_json = sorted((metrics_path).glob('metrics*.json'))    
                    if len(train_json) == 0: train_json = sorted((metrics_path/'training').glob('metrics*.json'))
                    if len(train_json) != 1: raise ValueError('The JSON file containing the training results could not be found.')
                    else: train_json = train_json[-1]
                        
                    # 훈련의 마지막 epoch 정보를 가져온다.
                    latest_txt = sorted(Path(str(metrics_path).replace('output', 'models')).glob('latest.txt'))
                    if len(latest_txt) != 1: raise ValueError('The txt file containing the training epoch results could not be found.')
                    else:
                        with open(latest_txt[-1], "r") as f:
                            latest = f.readlines()[-1].strip().split('epoch')[-1]
                        if not latest.isnumeric(): raise ValueError(f'Warring: The latest epoch is unknown. -> {latest}')
                        latest = int(latest)
                    
                    # 테스트 정보를 모두 가져온다.
                    test_json = sorted((metrics_path/'test').glob(f'**/*{self.test_dirname}*/*{self.test_filename}*.json'))   
                    if len(test_json) >= 1:
                        test_epochs = []
                        for t in test_json:
                            test_epoch = str(t).split('epoch')[-1].split('/')[0]
                            # test_epoch = str(t).split(self.test_dirname)[-1].split('epoch')[-1].replace('/metrics-test', '').replace('.json', '')
                            # test_epoch = test_epoch.split('_')[0]
                            if not test_epoch.isnumeric(): raise ValueError(f'Warring: The testing epoch is unknown. -> {t.name}')
                            test_epochs.append(int(test_epoch))
                        use_test_json = OrderedDict()
                        for sort_index in np.argsort(test_epochs):
                            # use_test_json[test_epochs[sort_index]] = test_json[sort_index]
                            use_test_json[test_json[sort_index]] = test_epochs[sort_index]
                    else : raise ValueError('The JSON file containing the test results could not be found.')
                    metrics_dict[category][model][run_cate].update({run_id:{'train':train_json,'test':use_test_json, 'latest': latest}})
        category_list = list(metrics_dict.keys())
        use_cate = category_list[0]
        print(f'Category list : {len(category_list)}cnt -> {category_list}')
        model_list = list(metrics_dict[use_cate].keys())
        use_model = model_list[0]
        print(f'Model_list    : {len(model_list)}cnt -> {model_list}')
        run_list = list(metrics_dict[use_cate][use_model].keys())
        use_run = run_list[0]
        print(f'Run_list      : {len(run_list)}cnt -> {run_list}')
        for run_id, run_json in metrics_dict[use_cate][use_model][use_run].items():
            print(f'example run id: {run_id}')
            print(f'- latest epoch : {run_json["latest"]}')
            print(f'- train json file : {run_json["train"]}')
            for test_epoch, test_json in run_json['test'].items():
                print(f'- test json file : {test_epoch} -> {test_json}')
        return metrics_dict    
    # output 중, metric.json 경로 정보 셋팅하는 함수
    
    # 가져온 metric.json 기반으로 정보를 dataframe으로 변환 (테스트는 설정한 best_metric에 맞춰 높은 점수로 결정)
    def _json2df(self):
        sheet_data = self._set_a_df_info()
        df_dict = {s:'None' for s in self.sheet_list}
        
        for sheet, d_c in sheet_data.items():
            df = pd.DataFrame(data=d_c['data'], columns=d_c['col'][0])
            model_mean_time = []
            if 'runtime' in df.columns:
                df['runtime'] = pd.to_datetime(df['runtime']) #.dt.time
                for model_name in df['model'].unique():
                    mean_time = df[df['model']==model_name]['runtime'].mean().strftime('%H:%M:%S')
                    mean_m = mean_time.split(':')
                    mean_m = int(mean_m[0]) * 60 + int(mean_m[1])
                    model_mean_time.append([model_name, mean_time, mean_m])
                df['runtime'] = df['runtime'].dt.time            
            df_dict[sheet] = deepcopy(df)
            
            if model_mean_time != []:
                time_df = pd.DataFrame(data=model_mean_time, columns=['model', 'runtime (mean)', 'minutes (mean)']) 
                sheet_name =  str(sheet)+'_runtime'
                df_dict[sheet_name] = deepcopy(time_df)
        return df_dict          
    
    def _set_a_df_info(self):
        '''
        1. SAVE CONTENT - training   : run_id | loss | optim | lr | scheduler | DA | Sampling | batch | acc_step | model | real_epoch | run_time | metrics...
        2. SAVE CONTENT - validation : run_id | loss | optim | lr | scheduler | DA | Sampling | batch | acc_step | model | real_epoch | metrics...
        3. SAVE CONTENT - test       : run_id | loss | optim | lr | scheduler | DA | Sampling | batch | acc_step | model | real_epoch | latest | metrics...
        '''        
        basic_column_list = ['run_id', 'loss', 'optimizer', 'learning rate (lr)', 'lr scheduler', 'DA', 'Sampling', 'batch size', 'accumulation steps', 'model']
        data_dict = {s:{'data':[], 'col':[]} for s in self.sheet_list}

        for category, cate_dict in self.result_info.items():
            for model, model_dict in cate_dict.items():
                for run_cate, run_dict in model_dict.items():
                    for run_id, run_json in run_dict.items():
                        # 0. get a config
                        basic_data = self._read_df_config(str(run_json['train']).replace('output', 'models').replace('metrics.json', 'config.json').replace('/training', ''))
                        basic_data.insert(0, run_id)
                        # 1. get a train
                        tr, val, _ = self._read_df_result(run_json['train'])
                        data_dict['training']['data'].append(self._list_concat(basic_data, list(tr.values())))
                        data_dict['training']['col'].append(self._list_concat(basic_column_list, list(tr.keys())))
                        if len(list(val.keys())) != 1:
                            data_dict['validation']['data'].append(self._list_concat(basic_data, list(val.values())))
                            data_dict['validation']['col'].append(self._list_concat(basic_column_list, list(val.keys())))
                        # 2. get a test (Finding the highest score (TNR))
                        test_data, test_col = self._get_test_result(run_json, basic_data, basic_column_list)
                        data_dict['test']['data'].append(test_data)
                        data_dict['test']['col'].append(test_col)
        return data_dict
    
    def _get_test_result(self, run_json, basic_data, basic_column_list):
        # return data, col
        raise NotImplementedError
        
    def _read_df_result(self, json_path, mode='train', latest=None):
        if mode not in ['train', 'test']: TypeError('The model can only accept "train" and "test" as inputs.')
        json_content = self._read_json(json_path)
        train_epoch = len(json_content['epoch']) if mode == 'train' else 0
        # test_epoch = json_content['epoch'][0] if mode == 'test' else 0 #-> 나중에 이걸로 수정
        test_epoch = 0
        if mode == 'test': 
            test_epoch = json_content['epoch'][0] if type(json_content['epoch']) == list else json_content['epoch']
        train, valid, test = {'epoch':train_epoch}, {'epoch':train_epoch}, {'epoch':test_epoch, 'latest':latest}
        if mode == 'train': train = {'epoch':train_epoch, 'runtime':':'.join(json_content['totaltime'].split('.')[:-1])}
        
        for k, v in json_content.items():
            if k in ['epoch', 'runtime', 'totaltime', 'loss', 'confusion', 'val_loss', 'val_confusion']: continue
            if mode == 'train':
                if 'val' in k: valid[k.split('val_')[-1]] = v[-1]
                else: train[k] = v[-1]
            else: test[k] = v
        return train, valid, test

    def _read_df_config(self, json_path):
        json_content = self._read_json(json_path)
        optim = json_content['optimizer']['type']
        lr = json_content['optimizer']['args']['lr']
        scheduler = json_content['lr_scheduler']['type'] if 'lr_scheduler' in json_content.keys() else None
        model = json_content['arch']['type']
        batch = json_content['data_loader']['args']['batch_size']
        acc_step = json_content['trainer']['accumulation_steps'] if 'accumulation_steps' in json_content['trainer'].keys() else None
        loss = json_content['loss']
        da = json_content['data_augmentation']['type'] if 'data_augmentation' in json_content.keys() else None
        sampling =  json_content['data_sampling']['name'] if 'data_sampling' in json_content.keys() else None
        return [loss, optim, lr, scheduler, da, sampling, batch, acc_step, model]

    def _list_concat(self, one, two):
        data = deepcopy(one)
        data.extend(two)
        return data
    # 가져온 metric.json 기반으로 정보를 dataframe으로 변환 (테스트는 설정한 best_metric에 맞춰 높은 점수로 결정)

    
    # 변환한 dataframe을 기반으로 시각화 출력물 저장하기
    def _close_all_plots(self):    
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
        
    def save2excel(self, save_name:str=None, extension:str='xlsx'):
        # 0. SAVE FILE : self.output_dir / (self.name.extension or save_name.extension)
        save_name = save_name if save_name is not None else self.name
        save_name += f'.{extension}' 
        if str(save_name)[0] != '/': save_path = self.output_dir / save_name
        else: save_path = Path(save_name)
        if not save_path.parent.is_dir(): save_path.parent.mkdir(parents=True, exist_ok=True)
        
        writer=pd.ExcelWriter(save_path, engine='openpyxl')
        for sheet_name, df in self.df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name)
        writer.close()
    
    def save2bar_per_model(self, df, optimizer, loss, by_metric:str, by_metric_replace_str:str=None, by:list=['lr scheduler', 'DA', 'Sampling'], 
                           min_y:float=0.75, bar_width=0.8, figsize:tuple=(17, 5), colormap:str='tab20', xtick_rotation:int=0,
                           show_score:bool=True, show_legend:bool=True, show_result:bool=True,
                           save_name:str=None, extension:str='png', just_show:bool=False):
        # by에는 optimizer, loss를 제외한 구분 특성을 입력해야 합니다.
        save_name = save_name if save_name is not None else self.name
        save_name += f'.{extension}' 
        if str(save_name)[0] != '/': save_path = self.output_dir / save_name
        else: save_path = Path(save_name)
        if not save_path.parent.is_dir(): save_path.parent.mkdir(parents=True, exist_ok=True)
        
        title = (by_metric.lower().capitalize() if by_metric_replace_str is None else by_metric_replace_str) + ' by '
        for idx, b in enumerate(by, 1):
            title += b.lower().capitalize()
            if idx <= len(by)-2: title += ', '
            elif idx == len(by)-1: title += ' and '
            else: title += '.'
        
        index_list = ['model']
        index_list.extend(deepcopy(by))
        stack_list = deepcopy(by)
        choose_list = deepcopy(by)
        choose_list.append('model')
        choose_list.append(by_metric)
        
        need_info = df.loc[(df['optimizer']==optimizer) & (df['loss']==loss), choose_list]
        need_info = need_info.set_index(index_list)
        pivot_table = need_info[by_metric].unstack(stack_list)

        ax = pivot_table.plot(kind='bar', stacked=False, figsize=figsize, fontsize=10, colormap=colormap, ylim=[min_y, 1], width=bar_width)

        # 각 막대에 값 표시
        if show_score:
            for bar_container in ax.containers: # 막대 컨테이너 가져오기
                for bar, _ in zip(bar_container, pivot_table.columns):
                    height = bar.get_height()  # 막대의 높이 가져오기
                    padding = 0.05
                    val = str(round(height*100, 2)).replace('.00', '')
                    if val[-1] == '0' and '.' in val: val = val[:-2]
                    val = ''.join([h+'\n' for h in val])
                    if height-padding <= min_y: height = (min_y if height < min_y else height) + padding
                    ax.text(bar.get_x() + bar.get_width() / 2, height - padding, val, ha='center', va='bottom', fontsize=8)

        # 범례 설정
        if show_legend:
            ax.legend(title='Legend', loc='upper right', fontsize='small')  # 범례 위치 및 글꼴 크기 설정
            # 범례 항목 텍스트 변경
            new_labels = []
            for _ in pivot_table.columns:
                new_label = ''
                for idx, s in enumerate(stack_list):
                    new_value = 'None' if pd.isna(_[idx]) else _[idx]
                    new_label += f'{s} - {new_value}'
                    if idx < len(stack_list)-1: new_label += ' | '
                new_labels.append(new_label)
            ax.legend(labels=new_labels, loc='lower center', bbox_to_anchor=(0,-0.7,1,0.2), ncol=1)#, mode='expand')  # 범례 항목 텍스트 변경

        # x 축 눈금(label) 설정
        for tick in ax.get_xticklabels(): tick.set_rotation(xtick_rotation)

        # 그래프 제목 및 축 이름 설정
        fontsize, labelpad = 15, 15
        plt.title(f'\nOptim - {optimizer} & loss - {loss}', loc='right', fontsize=10)
        plt.title(title, fontsize=fontsize, pad=labelpad)
        plt.xlabel('Model', fontsize=fontsize, labelpad=labelpad)
        plt.ylabel(by_metric.lower().capitalize() if by_metric_replace_str is None else by_metric_replace_str, fontsize=fontsize, labelpad=labelpad)
        if not just_show: plt.savefig(save_path, bbox_inches="tight")
        else: return plt
        
        # 그래프 보여주기
        if show_result: plt.show()
        self._close_all_plots()
    # 변환한 dataframe을 기반으로 시각화 출력물 저장하기