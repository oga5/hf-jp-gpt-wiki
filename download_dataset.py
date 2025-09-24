from datasets import load_dataset

if __name__ == "__main__":
    # Wiki40B Japanese split example
    # This will download and cache the dataset via Hugging Face datasets
    ds = load_dataset("fujiki/wiki40b_ja")
    print(ds)
