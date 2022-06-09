import argparse
from os import listdir
from os.path import join
import torch
from tqdm import tqdm
from data.SPRSound.preprocess import preprocess
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
import collections
from torchaudio import load
from utils.util import write_json

def main(config):
    logger = config.get_logger('main')

    # retrieve main attributes
    task_level = config['main']['task_level']
    wav_dir = config['main']['wav_dir']
    out_file = config['main']['out_file']
    task = task_level//10
    level = task_level%10
    CLASSES = module_data.resp_classes(task, level)
    

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict, strict=False)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    output_log = {}
    with torch.no_grad():
        for wav_name in tqdm(listdir(wav_dir)):
            data, _ = load(join(wav_dir, wav_name))
            # TODO: enable GPU? (also see preprocess.py)
            data = data.squeeze().cpu().detach().numpy()
            processed = preprocess(data)
            processed = processed.to(device).unsqueeze(0)
            output = model(processed)
            pred = torch.argmax(output).item()
            output_log[wav_name] = CLASSES[pred]
    write_json(output_log, out_file)
    logger.info("Write output to {}".format(out_file))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # Custom arguments for main.py only
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')

    options = [
        CustomArgs(['--task'], type=int, target='main;task_level'),
        CustomArgs(['--wav'], type=str, target='main;wav_dir'),
        CustomArgs(['--out'], type=str, target='main;out_file'),
    ]

    config = ConfigParser.from_args(args, options=options)
    main(config)
