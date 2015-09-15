import decompound_annoy
from compound import Compound
from lattice import Lattice
import fileinput
import argparse

def lattice(c):
    return decompound_annoy.get_decompound_lattice(c, 250, 0.0)

def viterbi(c):
    return decompound_annoy.vit.viterbi_decode(Compound(c, None,
        Lattice(lattice(c))))

for line in fileinput.input():
    c = line.decode('utf8').strip()
    print " ".join(map(lambda p: "%d,%d" % p, viterbi(c)))

