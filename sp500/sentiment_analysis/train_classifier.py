import itertools


def create_full_dict():
    """
    Create a dictionary containing all possible combinations of compound, textblob, and label.

    This function generates a dictionary where each key is a tuple representing a combination
    of compound, textblob, and label options. The value for each key is initialized to 0.

    Returns:
        full_dit (dict): A dictionary containing all possible combinations of compound, textblob, and label,
        with values initialized to 0.
    """

    compound_options = [
        "Positive High",
        "Positive Low",
        "Negative Low",
        "Negative High",
        "Neutral",
    ]
    textblob_options = [
        "Positive High",
        "Positive Low",
        "Negative Low",
        "Negative High",
        "Neutral",
    ]
    label = ["P", "N", "U"]

    full_dict = {}

    for compound, textblob, label in itertools.product(
        compound_options, textblob_options, label
    ):
        full_dict[(compound, textblob, label)] = 0

    return full_dict


def train_classifier(labeled_dataset):
    """
    Train a classifier to determine the most frequent label for each combination of
    compound and textblob.

    This function trains a classifier by analyzing a labeled dataset containing segmentation
    compound, segmentation textblob, and label information. It creates a mapping dictionary
    where each key represents a combination of segmentation compound and segmentation textblob,
    and the corresponding value is the most frequent label.

    Returns:
        mapping_dict (dict): A mapping dictionary where each key is a tuple representing
        a combination of segmentation compound and segmentation textblob, and the value is
        the most frequent label for that combination.
    """

    full_dict = create_full_dict()
    df = labeled_dataset[
        ["Segmentation Compound", "Segmentation Textblob", "Label (P, N, U-neutral)"]
    ]

    # Create a new column containing the tuple of segmentation compound and
    # segmentation textblob, and label
    df = df.assign(
        mapping=df.apply(
            lambda row: (
                row["Segmentation Compound"],
                row["Segmentation Textblob"],
                row["Label (P, N, U-neutral)"],
            ),
            axis=1,
        )
    )

    for _, row in df.iterrows():
        full_dict[row["mapping"]] += 1

    max_values = {}
    for key, value in full_dict.items():
        segment_tuple = (key[0], key[1])
        if segment_tuple not in max_values or value > max_values[segment_tuple]:
            max_values[segment_tuple] = value

    # Create a mapping dictionary
    mapping_dict = {}
    for key, value in full_dict.items():
        if value == max_values[(key[0], key[1])]:
            mapping_dict[(key[0], key[1])] = key[2]

    return mapping_dict
