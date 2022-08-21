import importlib
import os

import torch
import torch.nn as nn

from pytorch3dunet.datasets.utils import get_test_loaders
from pytorch3dunet.unet3d import utils
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.model import get_model

logger = utils.get_logger('UNet3DPredict')


def main():
    # Load configuration
    config = load_config()

    # Create the model
    model = get_model(config['model'])

    # Load model state
    model_paths = config['model_path']
    if not isinstance(model_paths, list):
        model_paths = [model_paths]

    # import pdb; pdb.set_trace()
    for model_path in model_paths:
        logger.info(f'Loading model from {model_path}...')
        utils.load_checkpoint(model_path, model)
        # use DataParallel if more than 1 GPU available
        device = config['device']
        # if torch.cuda.device_count() > 1 and not device.type == 'cpu':
        #     model = nn.DataParallel(model)
        #     logger.info(f'Using {torch.cuda.device_count()} GPUs for prediction')

        logger.info(f"Sending the model to '{device}'")
        model = model.to(device)

        output_dir = config['loaders'].get('output_dir', None)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f'Saving predictions to: {output_dir}')

        # create predictor instance
        predictor = _get_predictor(model, output_dir, config)

        test_loader_list = []
        for test_loader in get_test_loaders(config):
            # run the model prediction on the test_loader and save the results in the output_dir
            # predictor(test_loader)
            test_loader_list.append(test_loader)

        eval_inv_reg_only = config["eval_inv_reg_only"]
        suffix = os.path.basename(model_path)
        if eval_inv_reg_only:
            # only evaluate inv transformed pred vs. original label
            logger.info(f"evaluate inv reg only")
            predictor.evaluate_inv_transformed(test_loader_list, suffix="{}_predictions".format(suffix))
        else:
            predictor(test_loader_list, suffix="{}_predictions".format(suffix))
        # import pdb; pdb.set_trace()
        # exit(-1)


if __name__ == '__main__':
    main()