import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import contractions
import sys

nltk.download("stopwords")


stop_words = set(stopwords.words("english"))
onomatope = ["mm","hmm","hmmm","um","uh","oh","mh","sh","um"]


def remove_short_and_stop_words(text, min_length):
    words = text.split()
    # if len < min_length or word in stop word, and not '_' in the word, we get rid of it
    filtered_words = [
        word
        for word in words
        if (
            len(word) > min_length - 1 and
            not word in stop_words
            and not word in onomatope
            )
        or "_" in word
    ]

    result_string = " ".join(filtered_words)
    return result_string

def cleaning(str):
    # to lowercase 
    #str = str.lower()

    # remove contractions (don't => do not)
    #str = contractions.fix(str)

    # remove all possessive 's
    # f1 =  0.5950364963503649 and 0.5932957579353307 looks like it doesn't help 
    #str = str.replace("'s", "")
    # remove all the <vocal sound>, <disfmarker>, etc...
    #str = re.sub(r"<[^>]+>", "", str)
    # remove all characters that aren't letters (there is already no number in the dataset)
    #str = re.sub("[^a-z_<>]", " ", str)
    # remove short words with length < 4 and stop words
    #str = remove_short_and_stop_words(str, 4)

    return str


def main():
    if len(sys.argv) != 2:
        print("Usage: python clean_csv.py <input_csv_file>")
        sys.exit(1)

    input_csv_file = sys.argv[1]
    dataframe = pd.read_csv(input_csv_file, sep=",", header="infer")
    dataframe["TEXT"] = dataframe["TEXT"].apply(cleaning)
    output_csv_file = "clean_" + input_csv_file
    dataframe.to_csv(output_csv_file, index=False)

    print(f"Cleaning completed. Cleaned data saved to {output_csv_file}")

if __name__ == "__main__":
    main()