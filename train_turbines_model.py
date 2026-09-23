"""Compatibility entry point: search, then validation and production training."""
from search_random_forest import main as search_main
from train_random_forest import main as train_main


def main():
    search_main()
    train_main()


if __name__ == "__main__":
    main()
