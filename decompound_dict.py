#!/usr/bin/python

import sys
import fileinput
import argparse

def load_dict(file):
    splits = {}
    with open(file) as f:
        for line in f:
            es = line.decode('utf8').rstrip('\n').split(" ")
            w = es[0]
            indices = map(lambda i: i.split(','), es[1:])

            splits[w] = []
            for from_, to, fug in indices:
                s, e = int(from_), int(to)
                # Don't use single character splits - just add to prev split
                if e - s == 1:
                    splits[w][-1][1] += 1
                else:
                    splits[w].append([s, e, fug])
    return splits


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Decompound words.')
    parser.add_argument('dict')
    parser.add_argument('--drop_fugenlaute', help='If this flag is set, Fugenlaute (infixes such as -s, -es) are dropped from the final words.', action='store_true')
    parser.add_argument('--lowercase', help='Lowercase all words.', action='store_true')
    parser.add_argument('--restore_case', help='Restore the case (words will take case of the original word).', default=True)
    args = parser.parse_args()

    splits = load_dict(args.dict)
    def split_word(w):
        if w in splits:

            w_split = []
            for from_, to, fug in splits[w]:
                if args.drop_fugenlaute:
                    wordpart = w[from_:to-len(fug)]
                else:
                    wordpart = w[from_:to]

                if args.lowercase:
                    wordpart = wordpart.lower()
                elif args.restore_case == True:
                    if w == w.title():
                        wordpart = wordpart.title()
                    elif w == w.upper():
                        wordpart = wordpart.upper()

                w_split.append(wordpart)
                    
            return u" ".join(w_split)
        else:
            return w

    for line in sys.stdin:
        print u" ".join(map(split_word, line.decode('utf-8').strip().split(" ")))

