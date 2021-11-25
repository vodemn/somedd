from typing import Iterable


def run_length_encoding(seq: Iterable):
    result = []
    counter = 1
    current_symbol = None

    for sym in seq:
        if current_symbol is None:
            current_symbol = sym
            continue
        
        if sym == current_symbol:
            counter = counter + 1
        else:
            result.append((current_symbol, counter))
            current_symbol = sym
            counter = 1
    result.append((current_symbol, counter))
    return result
