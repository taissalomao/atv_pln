import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def print_tokens(tk):
    tc = 1
    print('Total tokens: ', len(tk))
    for token in tk:
        print("Token #", tc, ":", token)
        tc += 1

def print_tokens_as_list(tk):
    token_list = []
    for token in tk:
        token_list.append(token)
    
    print('Total tokens: ', len(token_list))
    return token_list


with open('./tokenizacao/texto.txt', 'r', encoding='utf-8') as file:
    sample_text_en = file.read()


sentences = sent_tokenize(sample_text_en)
print("Sentences:")
print_tokens(sentences)
print("\n")

words = word_tokenize(sample_text_en)
#Individualmente
print("Words:")
print_tokens(words)
print("\n")

pos_tags = pos_tag(words)
print("POS Tags (first 5):")
print_tokens(pos_tags[:5])

#Lista
""" print("\nWords:")
word_list = print_tokens_as_list(words)
print(word_list)

print("\nPOS Tags (first 5):")
pos_list = print_tokens_as_list(pos_tags[:5])
print(pos_list) """
