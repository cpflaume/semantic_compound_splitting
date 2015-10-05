#import decompound_annoy
from compound import Compound
from IPython.display import Image
import graphviz as gv
from lattice import Lattice

#def lattice(c):
#    return decompound_annoy.get_decompound_lattice(c, 250, 0.0)

#def viterbi(c):
#    return decompound_annoy.vit.viterbi_decode(Compound(c, None,
#        Lattice(lattice(c))))

#def draw_lattice(c):
#    return draw_lattice2(c, lattice(c))

#def draw_viterbi(c):
#    return draw_lattice2(c, lattice(c), viterbi=viterbi(c))

def draw_lattice2(c, l, viterbi=set()):
    a = gv.Digraph(engine='neato')

    allnodes = sorted(list(l.keys()))
    allnodes.append(len(c))
    a.body.append("%s " % (" ".join([ str(key) for (i, key) in enumerate(allnodes) ]))) #"%s [ pos = \"%s, 0!\" ]; \n" % (key, i*3)

    red = set()
    for (key, v) in l.items():
        for (from_, to, label, rank, cosine) in v:
            lbl = "%s (%d, %.2f)" % (label,rank,cosine)
            style = {}

            if (from_, to) in viterbi and (from_, to) not in red:
                style.update({'color': 'red', 'fontcolor': 'red'})
                red.add((from_, to))
            a.edge(str(from_), str(to), lbl, style)

    apply_styles(a, styles)
    return a #.render("x.png", view=True)

styles = {
    'graph': {
        'label': '',
        'rankdir': 'TB',
        'overlap': 'false',
        'splines': 'true',
        'sep': '1',
        'esep': '0.5'
    },
    'nodes': {

    },
    'edges': {

    }
}

def apply_styles(graph, styles):
    graph.graph_attr.update(
        ('graph' in styles and styles['graph']) or {}
    )
    graph.node_attr.update(
        ('nodes' in styles and styles['nodes']) or {}
    )
    graph.edge_attr.update(
        ('edges' in styles and styles['edges']) or {}
    )
    return graph

lattice = {0: [(0, 12, 'Hauptbahnhof', 0, 1.0),
  (0, 5, u'Haupt', 3, 0.57081479),
  (0, 5, u'Haupt', 2, 0.57828569),
  (0, 5, u'Haupt', 227, 0.55057061),
  (0, 5, u'Haupt', 232, 0.61008978),
  (0, 5, u'Haupt', 1, 0.6191287)],
 5: [(5, 12, 'Bahnhof', 0, 1.0),
  (5, 9, u'Bahn', 14, 0.43744743),
  (5, 9, u'Bahn', 159, 0.37221277),
  (5, 9, u'Bahn', 2, 0.52165151)],
 9: [(9, 12, 'hof', 0, 1.0)]}

viterbiH = [(0, 5), (5, 9), (9, 12)]

draw_lattice2("Hauptbahnhof", lattice, viterbiH)

