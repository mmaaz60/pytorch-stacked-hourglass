import argparse
import os.path

import torch
import torch.backends.cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from stacked_hourglass import hg1, hg2, hg3, hg4, hg5, hg6, hg7, hg8
from stacked_hourglass.datasets.mpii import Mpii, print_mpii_validation_accuracy
from stacked_hourglass.train import do_validation_epoch
from stacked_hourglass import model as m


def main(args):
    print(f"\nModel: {args.arch}")
    # Set the N & M
    m.N = args.N
    m.M = args.M
    # Select the hardware device to use for inference.
    if torch.cuda.is_available():
        device = torch.device('cuda', torch.cuda.current_device())
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Disable gradient calculations.
    torch.set_grad_enabled(False)

    pretrained = not args.model_file

    if pretrained:
        print('No model weights file specified, using pretrained weights instead.')

    # Create the model, downloading pretrained weights if necessary.
    if args.arch == 'hg1':
        model = hg1(pretrained=pretrained)
    elif args.arch == 'hg2':
        model = hg2(pretrained=pretrained)
    elif args.arch == 'hg3':
        model = hg3(pretrained=False)
    elif args.arch == 'hg4':
        model = hg4(pretrained=False)
    elif args.arch == 'hg5':
        model = hg5(pretrained=False)
    elif args.arch == 'hg6':
        model = hg6(pretrained=False)
    elif args.arch == 'hg7':
        model = hg7(pretrained=False)
    elif args.arch == 'hg8':
        model = hg8(pretrained=False)
    else:
        raise Exception('unrecognised model architecture: ' + args.model)
    model = model.to(device)

    if not pretrained:
        assert os.path.isfile(args.model_file)
        print('Loading model weights from file: {}'.format(args.model_file))
        checkpoint = torch.load(args.model_file)
        state_dict = checkpoint['state_dict']
        if sorted(state_dict.keys())[0].startswith('module.'):
            model = DataParallel(model)
        model.load_state_dict(state_dict)

    # Initialise the MPII validation set dataloader.
    val_dataset = Mpii(args.image_path, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Generate predictions for the validation set.
    _, _, predictions = do_validation_epoch(val_loader, model, device, Mpii.DATA_INFO, args.flip,
                                            plot_predictions=args.visualize, folder_path=args.visualization_path)

    # Report PCKh for the predictions.
    print('\nFinal validation PCKh scores:\n')
    print_mpii_validation_accuracy(predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a stacked hourglass model.')
    parser.add_argument('--image-path', required=True, type=str,
                        help='path to MPII Human Pose images')
    parser.add_argument('--arch', metavar='ARCH', default='hg1',
                        choices=['hg1', 'hg2', 'hg3', 'hg4', 'hg5', 'hg6', 'hg7', 'hg8'],
                        help='model architecture')
    parser.add_argument('--model-file', default='', type=str, metavar='PATH',
                        help='path to saved model weights')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--batch-size', default=1, type=int, metavar='N',
                        help='batch size')
    parser.add_argument('--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--visualize', default=False, type=bool,
                        help='Either to visualize the predictions or not.')
    parser.add_argument('--visualization_path', default=None, type=str,
                        help='Directory path to save the plotted predictions.')
    parser.add_argument('--N', default=128, type=int,
                        help='No. of channels in earlier layers of Residual Block')
    parser.add_argument('--M', default=128, type=int,
                        help='No. of channels in the final layer of Residual Block')

    main(parser.parse_args())
