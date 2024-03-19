import string

def get_char_dict():
    
    with open("./dataset/utils/chars/GB2312.txt", "r", encoding="utf-8") as fp:
        chars = list(fp.readline())
    
    chars = [""] +  list(string.printable) + chars
    char_dict = {v:k for k,v in enumerate(chars)}

    return char_dict