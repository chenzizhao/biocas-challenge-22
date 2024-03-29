# BioCAS challenge submission

This is the source code our submission in response to [IEEE BioCAS 2022 Grand challenge on Respiratory Sound Classification](https://yongfu-li.github.io/biocas_contest_2022/contest.html).

## Set up for development

Step 1: The raw data are provided <https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound>, in the form of .wav and .json (as illustrated in `testcase`). Clone this code [repo](https://github.com/chenzizhao/biocas-challenge-22/) which contains the raw data, preprocessing script and our model.

Step 2: Initialize the dependency with `conda`. The main dependency is `torch` and `torchaudio`.

```shell
cd biocas-challenge-22/
conda create --name biocas --file environment.yml
conda activate biocas
```

Step 3: Preprocess data. This generates files ready to be read by `data/SPRSound/Datasets.py`, which is then consumed by `data_loader/data_loaders.py`.

```shell
cd data/SPRSound/
python preprocess.py # this should populate the data/SPRSound/processed/ folder
```

Step 4: Take a look at `config_task21.json` first and run the following. This will train a basic model for Task 2-1.

```shell
python train.py -c config_task21.json
```

Step 5: Run mini test cases as provided by the organizers.

```shell
bash testcase/test.sh
```

which is just a few lines in the form of

```shell
python3 main.py --task 11 --wav /path/to/task1_wav/ --out /path/to/task11_output.json
```

```shell
python test.py -c config_task21.json -r saved/models/Audio_Resp_21/0523_235733/model_best.pth
```

## Notes on training

Checkout a visual for how the data flows in this repository. [Drawing](https://docs.google.com/drawings/d/1aEftpkjiR3YyvFvlrXHQfq8i23ZswIuHN55NdQemVbM/edit?usp=sharing).

The main drivers are `train.py` and `test.py` in the root folder.
You may want to pass in hyper parameters in CLI, or alternatively modify the `config.json` file. You can modify `data/preprocess.py` (remember to regenerate) and tweak arch/loss/metric in the `model/` folder. More in [Usage](#usage) section. Particularly, you may find this useful:

```shell
python train.py -c config_task21.json
python test.py -c config_task21.json -r saved/model/..best../model_best.pth
```

You can also visualize and monitor progress with Tensorboard:

```shell
tensorboard --logdir saved/log/..model../
```

More [Tensorboard Visualization](#tensorboard-visualization)

Or use our Weigh and Bias project (send me your wandb login email to get an invitation).
Note that wandb will take an extra minute or so to upload stuff. So I recommend only enable wandb when you'd like to share with the team. When you are testing your model locally, I'd use tensorboard instead (the default).

```shell
python train.py -c config_task21.json --wandb True
```

We keep track of various metrics such as `loss`, `accuracy`, `sensitivity`, `specifity` and a combined `score`. The metric we care about is `score_task1` and `score_task2` for the validation split. For reference, past year winners have ~0.85 `score`. More data in the Benchmark Tab of the challenge website.

To add/update dependency, use:

```shell
conda install scipy
conda remove scipy
conda clean -a
conda env export > environment.yml
pip list --format=freeze > requirements.txt
```

## Release checklist (order matters)

1. Update conda dependency
2. Upload release versions to wandb
3. Update the saved_resume_path in main.py
4. Test in root folder `bash testcase/test.sh`
5. Copy everything to a release folder, note:
    * data/SPRSound/: keep only init.py, preprocess.py and Dataset.py.
    __Do not copy__ the wav or the clip folder, because they are huge.
    * saved: we only need to keep those used in main.py (Check list 1)
    * drop cache and wandb/
6. In the release folder, double check `bash testcase/test.sh` works.
7. Zip the release folder. 

## Acknowledegment

This repository is generated from the [PyTorch Template Project](https://github.com/victoresque/pytorch-template) by [Victor Huang](https://github.com/victoresque). Below is the original `README.md`, just for your reference.

# PyTorch Template Project

PyTorch deep learning project made easy.

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [PyTorch Template Project](#pytorch-template-project)
  * [Requirements](#requirements)
    * [Features](#features)
    * [Folder Structure](#folder-structure)
    * [Usage](#usage)
      * [Config file format](#config-file-format)
      * [Using config files](#using-config-files)
      * [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
    * [Customization](#customization)
      * [Custom CLI options](#custom-cli-options)
      * [Data Loader](#data-loader)
      * [Trainer](#trainer)
      * [Model](#model)
      * [Loss](#loss)
      * [metrics](#metrics)
      * [Additional logging](#additional-logging)
      * [Validation data](#validation-data)
      * [Checkpoints](#checkpoints)
    * [Tensorboard Visualization](#tensorboard-visualization)
    * [Contribution](#contribution)
    * [TODOs](#todos)
    * [License](#license)
    * [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements

* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))

## Features

* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.

## Folder Structure

  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── new_project.py - initialize new project with template files
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

## Usage

The code in this repo is an MNIST example of the template.
Try `python train.py -c config.json` to run code.

### Config file format

Config files are in `.json` format:

```javascript
{
  "name": "Mnist_LeNet",        // training session name
  "n_gpu": 1,                   // number of GPUs to use for training.
  
  "arch": {
    "type": "MnistModel",       // name of model architecture to train
    "args": {

    }                
  },
  "data_loader": {
    "type": "MnistDataLoader",         // selecting data loader
    "args":{
      "data_dir": "data/",             // dataset path
      "batch_size": 64,                // batch size
      "shuffle": true,                 // shuffle training data before splitting
      "validation_split": 0.1          // size of validation dataset. float(portion) or int(number of samples)
      "num_workers": 2,                // number of cpu processes to be used for data loading
    }
  },
  "optimizer": {
    "type": "Adam",
    "args":{
      "lr": 0.001,                     // learning rate
      "weight_decay": 0,               // (optional) weight decay
      "amsgrad": true
    }
  },
  "loss": "nll_loss",                  // loss
  "metrics": [
    "accuracy", "top_k_acc"            // list of metrics to evaluate
  ],                         
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 100,                     // number of training epochs
    "save_dir": "saved/",              // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                    // save checkpoints every save_freq epochs
    "verbosity": 2,                    // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "min val_loss"          // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 10                  // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,               // enable tensorboard visualization
  }
}
```

Add addional configurations if you need.

### Using config files

Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints

You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU

You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.

  ```
  python train.py --device 2,3 -c config.json
  ```

  This is equivalent to

  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

## Customization

### Project initialization

Use the `new_project.py` script to make your new project directory with template files.
`python new_project.py ../NewProject` then a new project folder named 'NewProject' will be made.
This script will filter out unneccessary files like cache, git files or readme file.

### Custom CLI options

Changing values of config file is a clean, safe and easy way of tuning hyperparameters. However, sometimes
it is better to have command line options if some values need to be changed too often or quickly.

This template uses the configurations stored in the json file by default, but by registering custom options as follows
you can change some of them using CLI flags.

  ```python
  # simple class-like object having 3 attributes, `flags`, `type`, `target`.
  CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
  options = [
      CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
      CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
      # options added here can be modified by command line flags.
  ]
  ```

`target` argument should be sequence of keys, which are used to access that option in the config dict. In this example, `target`
for the learning rate option is `('optimizer', 'args', 'lr')` because `config['optimizer']['args']['lr']` points to the learning rate.
`python train.py -c config.json --bs 256` runs training with options given in `config.json` except for the `batch size`
which is increased to 256 by command line options.

### Data Loader

* **Writing your own data loader**

1. **Inherit ```BaseDataLoader```**

    `BaseDataLoader` is a subclass of `torch.utils.data.DataLoader`, you can use either of them.

    `BaseDataLoader` handles:
    * Generating next batch
    * Data shuffling
    * Generating validation data loader by calling
    `BaseDataLoader.split_validation()`

* **DataLoader Usage**

  `BaseDataLoader` is an iterator, to iterate through batches:

  ```python
  for batch_idx, (x_batch, y_batch) in data_loader:
      pass
  ```

* **Example**

  Please refer to `data_loader/data_loaders.py` for an MNIST data loading example.

### Trainer

* **Writing your own trainer**

1. **Inherit ```BaseTrainer```**

    `BaseTrainer` handles:
    * Training process logging
    * Checkpoint saving
    * Checkpoint resuming
    * Reconfigurable performance monitoring for saving current best model, and early stop training.
      * If config `monitor` is set to `max val_accuracy`, which means then the trainer will save a checkpoint `model_best.pth` when `validation accuracy` of epoch replaces current `maximum`.
      * If config `early_stop` is set, training will be automatically terminated when model performance does not improve for given number of epochs. This feature can be turned off by passing 0 to the `early_stop` option, or just deleting the line of config.

2. **Implementing abstract methods**

    You need to implement `_train_epoch()` for your training process, if you need validation then you can implement `_valid_epoch()` as in `trainer/trainer.py`

* **Example**

  Please refer to `trainer/trainer.py` for MNIST training.

* **Iteration-based training**

  `Trainer.__init__` takes an optional argument, `len_epoch` which controls number of batches(steps) in each epoch.

### Model

* **Writing your own model**

1. **Inherit `BaseModel`**

    `BaseModel` handles:
    * Inherited from `torch.nn.Module`
    * `__str__`: Modify native `print` function to prints the number of trainable parameters.

2. **Implementing abstract methods**

    Implement the foward pass method `forward()`

* **Example**

  Please refer to `model/model.py` for a LeNet example.

### Loss

Custom loss functions can be implemented in 'model/loss.py'. Use them by changing the name given in "loss" in config file, to corresponding name.

### Metrics

Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:

  ```json
  "metrics": ["accuracy", "top_k_acc"],
  ```

### Additional logging

If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log.update(additional_log)
  return log
  ```

### Testing

You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data

To split validation data from a data loader, call `BaseDataLoader.split_validation()`, then it will return a data loader for validation of size specified in your config file.
The `validation_split` can be a ratio of validation set per total data(0.0 <= float < 1.0), or the number of samples (0 <= int < `n_total_samples`).

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints

You can specify the name of the training session in config files:

  ```json
  "name": "MNIST_LeNet",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:

  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.mnt_best,
    'config': self.config
  }
  ```

### Tensorboard Visualization

This template supports Tensorboard visualization by using either  `torch.utils.tensorboard` or [TensorboardX](https://github.com/lanpa/tensorboardX).

1. **Install**

    If you are using pytorch 1.1 or higher, install tensorboard by 'pip install tensorboard>=1.14.0'.

    Otherwise, you should install tensorboardx. Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training**

    Make sure that `tensorboard` option in the config file is turned on.

    ```
     "tensorboard" : true
    ```

3. **Open Tensorboard server**

    Type `tensorboard --logdir saved/log/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images, and histogram of model parameters will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` and `torch.utils.tensorboard.SummaryWriter` modules.

**Note**: You don't have to specify current steps, since `WriterTensorboard` class defined at `logger/visualization.py` will track current steps.

## Contribution

Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs

* [ ] Multiple optimizers
* [ ] Support more tensorboard functions
* [x] Using fixed random seed
* [x] Support pytorch native tensorboard
* [x] `tensorboardX` logger support
* [x] Configurable logging layout, checkpoint naming
* [x] Iteration-based training (instead of epoch-based)
* [x] Adding command line option for fine-tuning

## License

This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgements

This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
