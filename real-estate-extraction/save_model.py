from model import ner_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hp_path', type=str)
parser.add_argument('--weight_path', type=str)
parser.add_argument('--outdir', type=str)
parser.add_argument('--version', type=int, default=1)
args = parser.parse_args()

ner_model.load_and_save_model(
    hp_path=args.hp_path, weight_path=args.weight_path, out_path=args.outdir, version=args.version
)
