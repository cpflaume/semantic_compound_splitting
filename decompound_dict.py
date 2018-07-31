#!/usr/bin/python

import sys
import fileinput
import argparse

def load_dict(file, ignore_case=False):
    splits = {}
    with open(file) as f:
        for line in f:
            es = line.rstrip('\n').split(" ")
            w = es[0]
            if args.ignore_case:
                # TODO, always using the last one in case of overlaps
                w = w.lower()
            
            indices = [i.split(',') for i in es[1:]]

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
    parser.add_argument('--lowercase', help='Lowercase all output words.', action='store_true')
    parser.add_argument('--ignore_case', help='Ignore upper/lowercase (words passed should be all lowercase)', action='store_true', default=False)
    parser.add_argument('--restore_case', help='Restore the case (words will take case of the original word).', default=True)
    args = parser.parse_args()

    splits = load_dict(args.dict, ignore_case=args.ignore_case)
    def split_word(w):
        if args.ignore_case:
            w = w.lower()
        
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
                    
            return " ".join(w_split)
        else:
            return w

    for line in sys.stdin:
        print(" ".join(map(split_word, line.strip().split(" "))))

