from .architectures.baseline import BaselineFraudNN
import argparse

ARCHITECTURES = {"baseline": BaselineFraudNN, "wide": None}


def create_architecture(arch, **kwargs):
    try:
        return ARCHITECTURES[arch](**kwargs)
    except KeyError:
        raise ValueError(
            f"Haven't implemented {arch} yet. Available: {list_architectures()}"
        )


def list_architectures():
    return list(ARCHITECTURES.keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs=1, help="Architecture argument")

    args = parser.parse_args()
    # print(args.arch[0])
    if args.arch:
        try:
            create_architecture(args.arch[0], pos_weight=258)
        except:
            print("doesn't work")
