from gingerit.gingerit import GingerIt

def correct_sentence(sentence):
    parser = GingerIt()
    result = parser.parse(sentence)
    corrected_sentence = result['result']
    return corrected_sentence

# Example usage
sentence = "I havv a bok."
corrected_sentence = correct_sentence(sentence)
print(corrected_sentence)
