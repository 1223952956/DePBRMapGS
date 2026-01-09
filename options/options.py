import yaml
from easydict import EasyDict as edict


def add_arguments(parser):
    """
    need a predefined mesh, uv_mapping, camera poses, camera intrinsics, camera images, (optional: normal maps)
    """
    parser.add_argument("-o", "--option", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-r", "--rootpath", type=str)
    parser.add_argument("-s", "--scene", type=str)
    parser.add_argument("-c", "--cuda", type=int)
    parser.add_argument("-e", "--exp", type=str)
    parser.add_argument("-g", "--group", type=str)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--app", type=str)
    parser.add_argument("--other_path", type=str)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--pose-idx", type=int)


def load_yaml_options(args):
    with open(f"options/{args.option}.yaml", "r") as file:
        options = edict(yaml.safe_load(file))
    for key, value in vars(args).items():
        if value is not None:
            options[key] = value
    return options
