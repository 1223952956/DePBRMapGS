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
    parser.add_argument(
        "--appearance-mode",
        choices=("rgb", "sh", "pbr"),
        default=None,
        help="Appearance representation. Overrides the legacy YAML 'sh' option.",
    )


def load_yaml_options(args):
    with open(f"options/{args.option}.yaml", "r") as file:
        options = edict(yaml.safe_load(file))
    for key, value in vars(args).items():
        if value is not None:
            options[key] = value

    # Backward compatibility: existing configs only have the boolean `sh` option.
    # New configs and CLI calls should prefer `appearance_mode`.
    if "appearance_mode" not in options:
        options.appearance_mode = "sh" if options.get("sh", False) else "rgb"
    options.appearance_mode = options.appearance_mode.lower()
    valid_appearance_modes = {"rgb", "sh", "pbr"}
    if options.appearance_mode not in valid_appearance_modes:
        raise ValueError(
            f"Unknown appearance_mode '{options.appearance_mode}'. "
            f"Expected one of {sorted(valid_appearance_modes)}."
        )

    # Keep old call sites working while appearance_mode is adopted throughout
    # the project. PBR is deliberately not treated as SH.
    options.sh = options.appearance_mode == "sh"
    return options
