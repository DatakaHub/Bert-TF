import argparse
from src.bert_urls import map_name_to_handle

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=list(map_name_to_handle.keys()) + ["all"],
        default="all",
        help="model name (see bert_urls.py)",
    )
    parser.add_argument(
        "--saveh5",
        nargs="?",
        type=str,
        default="./converted.h5",
        help="where to save the model",
    )
    args = parser.parse_args()
    from src.loader import load_bert_model
    from src.extract_all import retrieve_hub_data, all_extract

    if args.model_name == "all":
        all_extract()
    else:
        load_bert_model(
            weights_path=retrieve_hub_data(bert_model_name=args.model_name),
            save_path=args.saveh5,
        )
