# Harakat
FATHA = "\u064e"
DAMMA = "\u064f"
KASRA = "\u0650"
FATHATAN = "\u064b"
DAMMATAN = "\u064c"
KASRATAN = "\u064d"
SHADDA = "\u0651"
SUKUN = "\u0652"

HARAKAT = [FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SHADDA, SUKUN]

# Arabic letters
AR_LETTERS = "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي"

# Punctuations
PUNCTUATIONS = [".", "،", ":", "؛", "-", "؟", ","]

# Valid Arabic characters
VALID_ARABIC = HARAKAT + list(AR_LETTERS) + PUNCTUATIONS


def get_diac_vocabulary():
    vocab = ["", "[UNK]", " "]
    vocab += HARAKAT

    tashkeel = set(HARAKAT) - set([SHADDA, SUKUN])
    for diac in tashkeel:
        vocab.append(SHADDA + diac)

    return vocab


def get_letters_vocabulary():
    vocab = ["", "[UNK]", " ", "s", "e"]
    vocab += list(AR_LETTERS)
    return vocab
