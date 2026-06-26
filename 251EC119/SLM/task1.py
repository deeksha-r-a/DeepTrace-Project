# Load
dataset = load_dataset("roneneldan/TinyStories")

# Print dataset information
print(dataset)

# Print first 3 stories from the training split
for i in range(3):
    print(f"Story {i+1}: ")
    print(dataset['train'][i])
    print("\n")


# Next token prediction:
# Next-token prediction is how a language model learns by guessing the next piece of text (called a token) based on the text that came before it.
# During training, it makes billions of these guesses and adjusts itself whenever it's wrong. Over time, this helps it learn grammar, facts, reasoning patterns, and writing styles.
# When you chat with it, it generates responses by predicting one token at a time until the answer is complete.
