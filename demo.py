import decompound_annoy
from compound import Compound
from IPython.display import Image
import graphviz as gv
from lattice import Lattice

def lattice(c):
    return decompound_annoy.get_decompound_lattice(c, 250, 0.0)

def viterbi(c):
    return decompound_annoy.vit.viterbi_decode(Compound(c, None,
        Lattice(lattice(c))))

def draw_lattice(c):
    return draw_lattice2(c, lattice(c))

def draw_viterbi(c):
    return draw_lattice2(c, lattice(c), viterbi=viterbi(c))

def draw_lattice2(c, l, viterbi=set()):
    a = gv.Digraph()

    a.body.append("subgraph {rank=same; %s }" % (" ".join([ str(key) for key in l ] + [str(len(c))])))
    red = set()
    for (key, v) in l.items():
        for (from_, to, label, rank, cosine) in v:
            style = {}
            
            if (from_, to) in viterbi and (from_, to) not in red:
                style = {'color': 'red', 'fontcolor': 'red'}
                red.add((from_, to))


            a.edge(str(from_), str(to), "%s (%d, %.2f)" % (label,rank,cosine), style)

    apply_styles(a, styles)
    return a #.render("x.png", view=True)

styles = {
    'graph': {
        'label': '',
        'rankdir': 'TB',
        'overlap': 'false',
        'splines': 'true'
    },
    'nodes': {

    },
    'edges': {
        'labeldistance': '10'
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

vwl = {0: [(0, 21, 'Volkswirtschaftslehre', 0, 1.0),
  (0, 5, u'Volks', 12, 0.45278138),
  (0, 5, u'Volks', 161, 0.39893898)],
 5: [(5, 21, 'Wirtschaftslehre', 0, 1.0),
  (5, 16, u'Wirtschafts', 155, 0.39184004),
  (5, 16, u'Wirtschafts', 80, 0.38657355)],
 16: [(16, 21, 'Lehre', 0, 1.0)]}

draw_lattice2("Volkswirtschaftslehre", vwl, [ (16, 21) ])

