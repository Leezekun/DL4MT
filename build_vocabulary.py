from collections import OrderedDict
import sys
import numpy as np
import json

def build_vocabulary():
    word_freqs = OrderedDict()
    for filename in sys.argv[1:]:
        print('Processing', filename)
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

    with open('./data/joint_vocab_freq.json', 'w', encoding='utf-8') as f:
        json.dump(word_freqs, f, indent=2, ensure_ascii=False)

    # summarize the low-frequently tokens
    
    threshold = [1, 5, 10, 50, 100]
    num_filtered = [0]*len(threshold)
    num_total = 0
    for word, freq in word_freqs.items():
        for i, t in enumerate(threshold):
            if freq <= t:
                num_filtered[i] += 1
        num_total += 1
    s = [f"total:{num_total}"]
    for i in range(len(threshold)):
        s.append(f"less than {threshold[i]}:{num_filtered[i]}")
    print(",".join(s))
    
    # filter out the low-frequently tokens
    threshold = 50
    filtered_word_freqs = OrderedDict()
    for word, freq in word_freqs.items():
        if freq > threshold:
         filtered_word_freqs[word] = freq
    print(f"the size of vocab after filtering: {len(filtered_word_freqs)}")

    # filtered_word_freqs = word_freqs

    # build dictionary
    words = list(filtered_word_freqs.keys())
    freqs = list(filtered_word_freqs.values())

    sorted_idx = np.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]


    worddict = OrderedDict()
    worddict['<EOS>'] = 0
    worddict['<GO>'] = 1
    worddict['<UNK>'] = 2
    worddict['<PAD>'] = 3
    # FIXME We shouldn't assume <EOS>, <GO>, and <UNK> aren't BPE subwords.
    for ii, ww in enumerate(sorted_words):
        worddict[ww] = ii+4

    # The JSON RFC requires that JSON text be represented using either
    # UTF-8, UTF-16, or UTF-32, with UTF-8 being recommended.
    # We use UTF-8 regardless of the user's locale settings.
    with open('./data/joint_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(worddict, f, indent=2, ensure_ascii=False)

    print('Done')

if __name__ == '__main__':
    build_vocabulary()