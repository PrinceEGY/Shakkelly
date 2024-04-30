from .constants import *

class Preprocessor:
    @classmethod
    def strip_tatweel(cls, text):
        return text.replace("ـ", "")

    @classmethod
    def collapse_whitespace(cls, text):
        return " ".join(text.split())

    @classmethod
    def replace_newline(cls, text, new=" "):
        return text.replace("\n", new)

    @classmethod
    def remove_tashkeel(cls, text):
        for haraka in HARAKAT:
            text = text.replace(haraka, "")
        return text
    
    @classmethod
    def strip_tashkeel(cls, text):
        stripped = Preprocessor.remove_tashkeel(text)
        text = text+" " # add space to avoid index out of range
        tashkeel = []
        idx = 0
        while idx < len(text)-1:
            curr, next = text[idx], text[idx+1]
            if curr in HARAKAT:
                if curr == SHADDA:
                    if next in HARAKAT:
                        tashkeel.append(curr+next)
                        idx+=1
                    else:
                        tashkeel.append(curr)
                else:
                    if next == SHADDA:
                        tashkeel.append(next+curr)
                        idx+=1
                    else:
                        tashkeel.append(curr)
            elif curr == " ":
                tashkeel.append(" ")
            elif curr not in HARAKAT and next not in HARAKAT:
                tashkeel.append(" ")
            idx+=1
        return list(stripped), tashkeel
    
    @classmethod
    def combine_tashkeel(cls, text, tashkeel):
        assert len(text) == len(tashkeel), "Length of text and tashkeel should be the same"
        result = []
        for i in range(len(text)):
            if text[i] == " ":
                result.append(text[i])
            else:
                if tashkeel[i] == " ":
                    result.append(text[i])
                else:
                    result.append(text[i]+tashkeel[i])
        return "".join(result)