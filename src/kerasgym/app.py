from __future__ import print_function
import argparse
import os
import sys
import logging as log
import importlib


def pars_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Keras gym - keep track of pre-trained models the easy way.",
        epilog="As an alternative to the commandline, params can be placed in a file, one per line, and specified on\
                the commandline like 'kers-gym @configfile.ext'.",
        fromfile_prefix_chars='@')
    parser.add_argument(
        "-p",
        "--path",
        help="path to model definitions and checkpoints",
        metavar="path")
    parser.add_argument(
        "-o",
        "--out",
        help="path to save output history and checkpoints",
        metavar="out")
    parser.add_argument(
        "-m",
        "--model",
        help="model name",
        metavar="model")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="epochs to train for",
        metavar="epochs")
    parser.add_argument(
        "-c",
        "--continue",
        help="continue training from previous checkpoint",
        action="store_true",
        dest="contd")
    parser.add_argument(
        "-f",
        "--force",
        help="force overwrite existing model/history",
        action="store_true")
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    parser.add_argument(
        "-d",
        "--debug",
        help="show debug messages",
        action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = pars_args()
    # Setup logging
    if args.verbose:
        loglevel = log.INFO
    else:
        loglevel = log.WARN
    if args.debug:
        loglevel = log.DEBUG
        args.verbose = True
    log.basicConfig(format="%(levelname)s: %(message)s", level=loglevel)
    log.debug("Args: %s", args)
    # import model
    if args.path:
        sys.path.append(args.path)
    else:
        args.path = ''
    if not args.out:
        args.out = args.path

    module = importlib.import_module(args.model)
    gymmodel = module.Model()

    model_file = os.path.join(args.path, args.model + '.h5')
    history_file = os.path.join(args.path, args.model + 'history.json')

    model_file_out = os.path.join(args.out, args.model + '.h5')
    history_file_out = os.path.join(args.out, args.model + 'history.json')

    # load saved model, or prepare to overwrite
    overwrite_warning = False
    if os.path.isfile(model_file):
        log.info('Found saved model: %s ...', model_file)
        if not args.force and not args.contd:
            overwrite_warning = True
        if args.force and not args.contd:
            log.warn('Saved model will be overwritten!')
        if args.contd:
            log.info('Loading saved model...')
            gymmodel.load_model(model_file)
    if not gymmodel.is_initialized:
        log.info('Initialising new model...')
        gymmodel.init_model()

    # load history
    if os.path.isfile(history_file):
        log.info('Found saved history: %s', history_file)
        if not args.force and not args.contd:
            overwrite_warning = True
        if args.force and not args.contd:
            log.warn('Saved history will be overwritten!')
        if args.contd:
            log.info('Loading saved history ...')
            gymmodel.load_history(history_file)
            log.info('%d epochs loaded', gymmodel._epoch)

    if overwrite_warning:
        log.info('Aborting; use -c to continue training, or -f to force overwrite.')
        return

    if args.verbose:
        gymmodel.summary()

    # train and save updated model
    gymmodel.train_update(epochs=args.epochs)
    gymmodel.save_model(model_file_out)
    log.info('Saved model to %s ...', model_file_out)
    gymmodel.save_history(history_file_out)
    log.info('Saved history to %s ...', history_file_out)

if __name__ == "__main__":
    main()
