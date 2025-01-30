from datasets import load_dataset

# English only
en = load_dataset("allenai/c4", "en")

# Other variants in english
en_noclean = load_dataset("allenai/c4", "en.noclean")
en_noblocklist = load_dataset("allenai/c4", "en.noblocklist")
realnewslike = load_dataset("allenai/c4", "realnewslike")