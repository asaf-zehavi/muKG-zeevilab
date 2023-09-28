import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname("__file__"),os.path.pardir), os.path.pardir)))
from src.py.args_handler import load_args
from src.py.load.kgs import read_kgs_from_folder
from src.py.model.general_models import kge_models

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
KG_TASK = 'lp'


def load_default_args(model_name: str):
    curPath = os.path.abspath(os.path.dirname(__file__))
    args = load_args(curPath + "/args_kge/" + model_name + r"_args.json")
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data',
                        type=str,
                        help='The directory containing the data (muKG format)')
    parser.add_argument('-m', '--model',
                        type=str,
                        help='The name of the model (must be implemented in muKG')
    parser.add_argument('--model-module',
                        type=str,
                        required=False,
                        default=None,
                        help='An alternative model module to be imported')
    parser.add_argument('--model-args',
                        type=str,
                        required=False,
                        default=None,
                        help='The path to the model\'s arguments. Default uses muKG\'s default args file.')
    parser.add_argument('--train',
                        action='store_true',
                        help='Should we train the model?')
    parser.add_argument('--valid',
                        action='store_true',
                        help='Should we run validation?')
    opts = parser.parse_args()

    model_name = opts.model
    if opts.model_args is not None:
        args = load_args(opts.model_args)
    else:
        args = load_default_args(model_name=model_name)

    args.training_data = opts.data
    kgs = read_kgs_from_folder(KG_TASK, args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                               remove_unlinked=False)

    model = kge_models(args, kgs)
    if opts.model_module is not None:
        model.get_model_explicit(model_name=args.embedding_module, model_module_path=opts.model_module)
    else:
        model.get_model(args.embedding_module)

    if opts.train:
        model.run()


if __name__ == '__main__':
    main()
