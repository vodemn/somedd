def run_length_encoding(seq):
    result = []
    counter = 1
    current_symbol = seq[0]
    for i in range(1, len(seq)):
        if seq[i] == current_symbol:
            counter = counter + 1
        else:
            result.append([current_symbol, counter])
            current_symbol = seq[i]
            counter = 1
    result.append([current_symbol, counter])
    return result
