# Config file

> Config files are in `.json` format.  
> You can use file `create_single_config.py` to generate a config file.    
> The following example is based on the examples provided in the repository.   
> Add addional configurations if you need.   

> ### **INDEX**
> - [Config file](#config-file)
>   * [`BaseRawDataLoader` Type](#baserawdataloader-type)
>   * [`BaseSplitDatasetLoader` Type](#basesplitdatasetloader-type)
> 
> <small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


## [`BaseRawDataLoader`](base/base_raw_data_loader.py) Type

```javascript
{
    "name": "MNIST",                  // Name for the output file.
    "n_gpu": 2,                       // number of GPUs to use for training.

    "arch": {
        "type": "LeNet5",             // Name of the model class. It must be defined in model.py.
        "args": {
            "num_classes": 10,
            "in_chans": 1,
            "H": 128,
            "W": 128
        },
        "visualization": true
    },
    "data_loader": {
        "type": "MnistDataLoader",     // Name of the data loader. It must be defined in the data_loader folder.
        "args":{
            "data_dir": "data/",       // dataset path
            "batch_size": 1024,        // batch size
            "shuffle": true,           // shuffle training data before splitting
            "validation_split": 0.1,   // size of validation dataset. float(portion) or int(number of samples)
            "num_workers": 2,          // number of cpu processes to be used for data loading      
            "trsfm": {                 // transforms
                "Resize": {
                    "size": 128
                },
                "ToTensor": null,
                "Normalize": {
                    "mean": [0.1307],
                    "std": [0.3081]
                }
            }     
        }
    },
    "data_augmentation": {
        "type": "Cutmix",              // The data augmentation technic defined in data_augmentation.py.
        "args": {
            "beta": 0.1,
            "prob": 0.5
        },
        "hook_args": {                 // Set up a hook on the layer where data augmentation will be performed.
            "layer_idx": 0,
            "pre": true                // Determine whether to apply data augmentation to the input values.
        }
    },
    "data_sampling":{                  // Perform data sampling for each batch.
        "type": "down",                // If "down," reduce the data. If "up," increase the data.
        "name": "random_downsampling"  // The data sampling technic defined in data_sampling.py.
    },
    "optimizer": {
        "type": "Adam",                // Optimizers supported by PyTorch. (https://pytorch.org/docs/stable/optim.html#algorithms)
        "args":{
            "lr": 0.001,               // learning rate
            "weight_decay": 0,         // (optional) weight decay
            "amsgrad": true
        }
    },
    "loss": "nll_loss",                // The loss function defined in loss.py.
    "metrics": {                       // list of metrics to evaluate using pycm object
        "ACC": null,                   // When calculating average metrics across all classes, it is null.
        "TPR": 1                       // When calculating metrics for a specific class, it is currently set to class 1 as an example.
    ,
    "curve_metrics":{                  // list of curve metrics to evaluate using scikit-learn
        "ROC": null,                   // Drawing a typical ROC curve for each class and using both micro and macro averaging methods.
        "FixedNegativeROC":{           // Calculating metrics based on a fixed goal (fixed_goal). Additionally, ROC curves will be drawn based on the class specified as negative.
            "save_dir": "fixedSpec",
            "negative_class_idx": 0,
            "fixed_goal": [0.95, 0.97, 0.99]
        }
    },
    "lr_scheduler": {                  // If you need a fixed learning rate, delete that part.
        "type": "StepLR",              // learning rate scheduler
        "args": {
            "step_size": 10,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 50,                   // number of training epochs
        "accumulation_steps": 4,        // gradient accumulation

        "save_dir": "saved/",           // checkpoints are saved in save_dir/models/name
        "save_period": 1,               // save checkpoints every save_freq epochs
        "verbosity": 2,                 // 0: quiet, 1: per epoch, 2: full
        
        "monitor": "min val_loss",      // mode and metric for model performance monitoring. set 'off' to disable.
        "early_stop": 10,               // number of epochs to wait before early stop. set 0 to disable.

        "tensorboard": true,            // enable tensorboard visualization
        "tensorboard_projector": {      // enable save projector at first epoch
            "train":false,
            "valid":true
        },
        "tensorboard_pred_plot": true,  // enable save prediction example plot per epoch (5 data)
        "save_performance_plot": true   // enable output diretory
    },
    "tester":{
        "tensorboard_projector": true   // enable save projector
    }
}
```
<br>

## [`BaseSplitDatasetLoader`](base/base_split_data_loader.py) Type

```javascript
{
    "name": "NPZ",                      // Name for the output file.
    "n_gpu": 2,                         // number of GPUs to use for training.

    "arch": {
        "type": "deit_tiny_patch16_LS", // Name of the model class. It must be defined in model.py.
        "args": {},
        "visualization": true
    },
    "data_loader": {
        "type": "NPZDataLoader",         // Name of the data loader. It must be defined in the data_loader folder.
        "args":{
            "dataset_path": "1.npz",     // dataset path
            "batch_size": 1024,          // batch size
            "mode":["train", "valid"],   // dataloader type list
            "num_workers": 2,            // number of cpu processes to be used for data loading      
            "trsfm": {                   // transforms
                "ToTensor": null,
                "SquarePad_Side": null,
                "Normalize": {
                    "mean": [0.1307],
                    "std": [0.3081]
                }
            }         
        }
    },
    "data_augmentation": {
        "type": "Cutmix",                // The data augmentation technic defined in data_augmentation.py.
        "args": {
            "beta": 0.1,
            "prob": 0.5
        },
        "hook_args": {                   // Set up a hook on the layer where data augmentation will be performed.
            "layer_idx": 0,
            "pre": true                  // Determine whether to apply data augmentation to the input values.
        }
    },
    "data_sampling":{                    // Perform data sampling for each batch.
        "type": "down",                  // If "down," reduce the data. If "up," increase the data.
        "name": "random_downsampling"    // The data sampling technic defined in data_sampling.py.
    },
    "optimizer": {
        "type": "Adam",                  // Optimizers supported by PyTorch. (https://pytorch.org/docs/stable/optim.html#algorithms)
        "args":{
            "lr": 0.001,                 // learning rate
            "weight_decay": 0,           // (optional) weight decay
            "amsgrad": true
        }
    }, 
    "loss": "bce_loss",                  // The loss function defined in loss.py.
    "metrics": {                         // list of metrics to evaluate using pycm object
        "ACC": null,                     // When calculating average metrics across all classes, it is null.
        "TPR": 1                         // When calculating metrics for a specific class, it is currently set to class 1 as an example.
    },
    "curve_metrics":{                    // list of curve metrics to evaluate using scikit-learn
        "ROC": null,                     // Drawing a typical ROC curve for each class and using both micro and macro averaging methods.
        "FixedNegativeROC":{             // Calculating metrics based on a fixed goal (fixed_goal). Additionally, ROC curves will be drawn based on the class specified as negative.
            "save_dir": "fixedSpec",
            "negative_class_idx": 0,
            "fixed_goal": [0.95, 0.97, 0.99]
        }
    }, 
    "lr_scheduler": {                    // If you need a fixed learning rate, delete that part.
        "type": "StepLR",                // learning rate scheduler
        "args": {
            "step_size": 10,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 50,                     // number of training epochs
        "accumulation_steps": 4,          // gradient accumulation

        "save_dir": "saved/",             // checkpoints are saved in save_dir/models/name
        "save_period": 1,                 // save checkpoints every save_freq epochs
        "verbosity": 2,                   // 0: quiet, 1: per epoch, 2: full
        
        "monitor": "min val_loss",        // mode and metric for model performance monitoring. set 'off' to disable.
        "early_stop": 10,                 // number of epochs to wait before early stop. set 0 to disable.

        "tensorboard": true,              // enable tensorboard visualization
        "tensorboard_projector": {        // enable save projector at first epoch
            "train":false,
            "valid":true
        },
        "tensorboard_pred_plot": true,    // enable save prediction example plot per epoch (5 data)
        "save_performance_plot": true     // enable output diretory
    },
    "tester":{
        "tensorboard_projector": true     // enable save projector
    }
}
```