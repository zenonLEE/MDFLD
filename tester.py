import os
import inspect
import torch
from lib import utility


def _load_checkpoint(net, config, args):
    model_path = args.pretrained_weight or os.path.join(config.model_dir, "best_model.pkl")
    load_options = {"map_location": config.device}
    # Original training checkpoints include non-tensor metadata (e.g. Counter).
    # Preserve the legacy loader on new PyTorch; use only trusted .pkl files.
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = False
    checkpoint = torch.load(model_path, **load_options)
    net.load_state_dict(checkpoint["net"])


def test(args):
    # conf
    config = utility.get_config(args)
    config.device_id = args.device_ids[0]
    config.crop_op = False
    config.aug_prob = 0
    # set environment
    utility.set_environment(config)
    #config.init_instance_test(os.path.join('./pretrained_model', args.data_definition) +'/')
    config.init_instance()
    if config.logger is not None:
        config.logger.info("Loaded configure file %s: %s" % (args.config_name, config.id))
        config.logger.info("\n" + "\n".join(["%s: %s" % item for item in config.__dict__.items()]))

    # model
    net = utility.get_net(config)
    _load_checkpoint(net, config, args)

    if config.logger is not None:
        config.logger.info("Loaded network")
        # config.logger.info('Net flops: {} G, params: {} MB'.format(flops/1e9, params/1e6))

    # data - test
    test_loader = utility.get_dataloader(config, "test")

    if config.logger is not None:
        config.logger.info("Loaded data from {:}".format(config.test_tsv_file))

    # inference
    result, metrics = utility.forward_test(config, test_loader, net)
    if config.logger is not None:
        config.logger.info("Finished inference")

    # output
    for k, metric in enumerate(metrics):
        if config.logger is not None and len(metric) != 0:
            config.logger.info(
                "Tested {} dataset, the Size is {}, Metric: [NME {:.6f}, FR {:.6f}, AUC {:.6f}]".format(
                    config.type, len(test_loader.dataset), metric[0], metric[1], metric[2]))

def test_img_only(args):
    # conf
    config = utility.get_config(args)
    config.device_id = args.device_ids[0]
    config.crop_op = False
    config.aug_prob = 0
    # set environment
    utility.set_environment(config)
    #config.init_instance_test(os.path.join('./pretrained_model', args.data_definition) +'/')
    config.init_instance()
    if config.logger is not None:
        config.logger.info("Loaded configure file %s: %s" % (args.config_name, config.id))
        config.logger.info("\n" + "\n".join(["%s: %s" % item for item in config.__dict__.items()]))

    # Validate the image input before loading the checkpoint.
    test_loader = utility.get_dataloader(config, "test_image")

    # model
    net = utility.get_net(config)
    _load_checkpoint(net, config, args)


    if config.logger is not None:
        config.logger.info("Loaded network")
        # config.logger.info('Net flops: {} G, params: {} MB'.format(flops/1e9, params/1e6))

    if config.logger is not None:
        config.logger.info("Loaded data from {:}".format(config.test_tsv_file))

    # inference
    return utility.forward_test_image_only(config, test_loader, net)


