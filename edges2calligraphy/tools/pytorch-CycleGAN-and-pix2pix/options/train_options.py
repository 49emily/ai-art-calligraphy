from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument(
            "--display_freq",
            type=int,
            default=400,
            help="frequency of showing training results on screen",
        )
        parser.add_argument(
            "--display_ncols",
            type=int,
            default=4,
            help="if positive, display all images in a single visdom web panel with certain number of images per row.",
        )
        parser.add_argument(
            "--display_id", type=int, default=1, help="window id of the web display"
        )
        parser.add_argument(
            "--display_server",
            type=str,
            default="http://localhost",
            help="visdom server of the web display",
        )
        parser.add_argument(
            "--display_env",
            type=str,
            default="main",
            help='visdom display environment name (default is "main")',
        )
        parser.add_argument(
            "--display_port",
            type=int,
            default=8097,
            help="visdom port of the web display",
        )
        parser.add_argument(
            "--update_html_freq",
            type=int,
            default=1000,
            help="frequency of saving training results to html",
        )
        parser.add_argument(
            "--print_freq",
            type=int,
            default=100,
            help="frequency of showing training results on console",
        )
        parser.add_argument(
            "--no_html",
            action="store_true",
            help="do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/",
        )
        # network saving and loading parameters
        parser.add_argument(
            "--save_latest_freq",
            type=int,
            default=5000,
            help="frequency of saving the latest results",
        )
        parser.add_argument(
            "--save_epoch_freq",
            type=int,
            default=5,
            help="frequency of saving checkpoints at the end of epochs",
        )
        parser.add_argument(
            "--save_by_iter",
            action="store_true",
            help="whether saves model by iteration",
        )
        parser.add_argument(
            "--continue_train",
            action="store_true",
            help="continue training: load the latest model",
        )
        parser.add_argument(
            "--epoch_count",
            type=int,
            default=1,
            help="the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...",
        )
        parser.add_argument(
            "--phase", type=str, default="train", help="train, val, test, etc"
        )
        # training parameters
        parser.add_argument(
            "--n_epochs",
            type=int,
            default=100,
            help="number of epochs with the initial learning rate",
        )
        parser.add_argument(
            "--n_epochs_decay",
            type=int,
            default=100,
            help="number of epochs to linearly decay learning rate to zero",
        )
        parser.add_argument(
            "--beta1", type=float, default=0.5, help="momentum term of adam"
        )
        parser.add_argument(
            "--lr", type=float, default=0.0002, help="initial learning rate for adam"
        )
        parser.add_argument(
            "--gan_mode",
            type=str,
            default="lsgan",
            help="the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.",
        )
        parser.add_argument(
            "--pool_size",
            type=int,
            default=50,
            help="the size of image buffer that stores previously generated images",
        )
        parser.add_argument(
            "--lr_policy",
            type=str,
            default="linear",
            help="learning rate policy. [linear | step | plateau | cosine]",
        )
        parser.add_argument(
            "--lr_decay_iters",
            type=int,
            default=50,
            help="multiply by a gamma every lr_decay_iters iterations",
        )

        self.isTrain = True
        return parser

    def parse(self, args=None):
        """Parse our options, create checkpoints directory suffix, and set up gpu device.

        If args is not None, it will parse the provided list of arguments instead of sys.argv
        """
        if args is not None:
            # Override the BaseOptions.parse() method to accept a list of arguments
            opt = self.gather_options_from_list(args)
        else:
            # Use the original parsing method
            opt = super().parse()

        return opt

    def gather_options_from_list(self, args_list):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.

        This version accepts a list of arguments instead of using sys.argv
        """
        if not self.initialized:  # check if it has been initialized
            parser = self.initialize_parser()

        # Get the basic options
        opt, _ = parser.parse_known_args(args_list)

        # Modify model-related parser options
        model_name = opt.model
        model_option_setter = self.get_model_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args(args_list)  # parse again with new defaults

        # Modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = self.get_dataset_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # Save and return the parser
        self.parser = parser
        opt = parser.parse_args(args_list)
        opt.isTrain = self.isTrain  # train or test

        # Process opt.suffix
        self.process_suffix(opt)
        self.print_options(opt)
        self.setup_gpu(opt)

        self.opt = opt
        return self.opt

    def initialize_parser(self):
        """Helper method to initialize the parser only once"""
        import argparse

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser = self.initialize(parser)
        return parser

    def get_model_option_setter(self, model_name):
        """Helper method to get the model option setter"""
        import models

        return models.get_option_setter(model_name)

    def get_dataset_option_setter(self, dataset_name):
        """Helper method to get the dataset option setter"""
        import data

        return data.get_option_setter(dataset_name)

    def process_suffix(self, opt):
        """Process the suffix to append to the name"""
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

    def setup_gpu(self, opt):
        """Set up GPU device based on gpu_ids"""
        import torch

        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
