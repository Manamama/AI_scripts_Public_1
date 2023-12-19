# Import the SentencePiece library
import sentencepiece as spm

# Open the text file
text_file = "input.txt" # Change this to your text file name
with open(text_file, "r", encoding="utf-8") as f:
    text = f.read()

# Write the text to a new file with one sentence per line
with open('sentences.txt', 'w', encoding="utf-8") as f:
    f.write(text.replace('. ', '.\n'))  # Replace periods followed by a space with a period and a newline

# Train SentencePiece model with Unigram
spm.SentencePieceTrainer.train('--input=sentences.txt --model_prefix=m --vocab_size=800')
#Unigram Language Model: This model tends to break down words into smaller subword units. For example, in your Unigram result, the word “All” is broken down into two tokens: ‘▁A’ and ‘ll’. This is because the Unigram model calculates the probability of each possible subword and chooses the most likely segmentation.
#
# Or: Train SentencePiece model with BPE
#spm.SentencePieceTrainer.train('--input=sentences.txt --model_prefix=m --vocab_size=800 --model_type=bpe')
#Byte Pair Encoding (BPE): This model, on the other hand, prefers to keep frequent words or subwords intact as much as possible. In your BPE result, the word “All” is kept as a single token: ‘▁All’. BPE starts with a base vocabulary of individual characters and iteratively merges the most frequent pair of symbols to create a new symbol, adding it to the vocabulary.

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load('m.model')

# Encode the text into subword pieces
pieces = sp.encode(text, out_type=str)

# Print the pieces
print(pieces)

# Decode the pieces back to the original text
text = sp.decode(pieces)

# Print the text
print(text)
