from synthetic.init import set_seed, read_config
from synthetic.generator import SyntheticData
from synthetic.functions import CreateFunctions


def main(cfg):
    set_seed(cfg.seed)

    # Generate functions
    generator = CreateFunctions(cfg)
    composed_functions, info = generator.compose()

    # Generate data using functions
    synData = SyntheticData(cfg, composed_functions, info)
    synData.init_tokens()
    corpus, _ = synData.generate_corpus()
    synData.store_data()

    print("\nExample data point:", synData.decode(corpus[0]))


if __name__ == "__main__":
    # Set config in the yaml files
    cfg = read_config("./config/gen/conf.yaml")
    main(cfg)
