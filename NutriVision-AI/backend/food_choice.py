def food_choice(food101_name, open_vocab_names):
    """Return a single suggestion only when the model names do not conflict."""
    if not open_vocab_names:
        return food101_name, [food101_name]

    def normalized(name):
        return " ".join(name.replace("_", " ").casefold().split())

    open_name = open_vocab_names[0]
    if normalized(food101_name) == normalized(open_name):
        return open_name, [open_name]
    return None, [food101_name, open_name]
