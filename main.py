# TODO

print("todo: pending something similar to test.py")
# exit(0)


import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import collections

from utils.util import write_json


def main(config):
    logger = config.get_logger('main')

    # retrieve main attributes
    task_level = config['main']['task_level']
    wav_dir = config['main']['wav_dir']
    out_file = config['main']['out_file']
    assert task_level in (11,12,21,22), "Task level must be in (11,12,21,22)"
    

    # setup data_loader instances
    data_loader = module_data.MainDataLoader(
        wav_dir,
        batch_size=1,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # TODO 
            output_log = output

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    # TODO same to output
    write_json(output_log, out_file)
    logger.info("Write output to {}".format(out_file))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-c', '--config', default='config_resp.json', type=str,
                      help='config file path (default: config_resp.json)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # Custom arguments for main.py only
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')

    options = [
        CustomArgs(['--task'], type=int, target='main;task_level'),
        # choices = [11, 12, 21, 22]
        CustomArgs(['--wav'], type=str, target='main;wav_dir'),
        # default = './testcase/task1_wav/'
        CustomArgs(['--out'], type=str, target='main;out_file'),
        # default = './testcase/my_output/task11_output.json'
    ]

    # Usage of this file 
    # python3 main.py --task 11 --wav testcase/task1_wav/ --out testcase/my_output/task11_output.json

    config = ConfigParser.from_args(args, options=options)
    main(config)
