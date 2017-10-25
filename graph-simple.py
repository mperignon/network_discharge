import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
import copy
import networkx as nx



edge_list = pd.read_csv('simple_alledges.csv')
node_list = pd.read_csv('simple_nodes_xy.csv')


## create a synthetic topography
relief = 26. # elev difference between highest point and ocean

d = node_list['y'].max() - node_list['y'].min()
node_list['elev'] = relief / d * (node_list['y'] - node_list['y'].min())

node_list['cat_color'] = 'b'
node_list['cat_color'][node_list['cat'] == 2] = 'r'
node_list['cat_color'][node_list['cat'] == 1] = 'g'



## add synthetic elevation values to edge_list

from_elev = []

for i in edge_list['START_pt']:
    e = node_list['elev'][node_list['NodeID'] == i].values[0]
    from_elev.append(e)
    
to_elev = []

for i in edge_list['END_pt']:
    e = node_list['elev'][node_list['NodeID'] == i].values[0]
    to_elev.append(e)
    
    
edge_list['from_elev'] = from_elev
edge_list['to_elev'] = to_elev

edge_list['elev_diff'] = edge_list['from_elev'] - edge_list['to_elev']
edge_list['slope'] = edge_list['elev_diff'] / edge_list['length']




g = nx.DiGraph()

# this assumes that the edge list has the correct directions
edge_ends = zip(edge_list['START_pt'], edge_list['END_pt'])  
g.add_edges_from(edge_ends)

for n in edge_list[edge_list.columns.values[3:]].iteritems():
    nx.set_edge_attributes(g, name=n[0], values = dict(zip(edge_ends, n[1])))


for n in node_list.iteritems():    
    nx.set_node_attributes(g, name = n[0], values = dict(zip(node_list['NodeID'], n[1])))
        
        
        
node_positions = dict()

for node in g.nodes(data=True):
    a = node[1] 
    if len(a)>0:
        x = a['x']
        y = a['y']
        node_positions[node[0]] = (x, y)

g.number_of_nodes()



sources = [u for u in g.nodes() if g.nodes[u]['cat'] == 1]
sinks = [u for u in g.nodes() if g.nodes[u]['cat'] == 2]


ys = np.array(nx.get_node_attributes(g,'y').values())
start = np.where(ys == ys.max())[0][0]

nodes_ds = list(g.out_edges(start))



# calculates the fractionation of flow out of each node
# uses only width
sinks_check = []
done = []

for this_node in g.nodes():

    nbrs = [n for n in g.successors(this_node)]

    if len(nbrs) > 0:
    
        widths = [g[this_node][n]['Width'] for n in nbrs]
        total_width = sum(widths)
        frac_widths = [n / total_width for n in widths]

        add_edges = [(this_node, nbrs[n], {'frac_w': frac_widths[n]}) for n in range(len(nbrs))]
        g.add_edges_from(add_edges)
        done.append(this_node)
        
    else:
        sinks_check.append(this_node)





Qs = [100,100,100,500,500] # discharge at the sources

# this adds the discharge values to the source nodes
add_nodes = [(sources[n], {'discharge': Qs[n]}) for n in range(len(sources))]
g.add_nodes_from(add_nodes)




node_order = list(nx.topological_sort(g))
add_nodes = []

for this_node in node_order:

    if len(g.in_edges(this_node)) == 0:
        # if this is a source node, take the discharge assigned above
        Q = g.nodes[this_node]['discharge']
        
    else:
        # if not a source node, then the discharge is the sum of the discharge at the incoming edges
        Q = sum([g[n[0]][n[1]]['discharge'] for n in g.in_edges(this_node)])
        g.add_nodes_from([(this_node, {'discharge': Q})])
        
    
    for nbr, eattr in g.adj[this_node].items():
        # once you have the discharge at the node, partition it to the outgoing edges
        g.add_edges_from([(this_node, nbr, {'discharge': Q * eattr['frac_w']})])
        



width = [g[u][v]['discharge']/150. for u,v in g.edges()]

source_x = [g.nodes[n]['x'] for n in sources]
source_y = [g.nodes[n]['y'] for n in sources]
sink_x = [g.nodes[n]['x'] for n in sinks]
sink_y = [g.nodes[n]['y'] for n in sinks]
    
plt.figure(figsize=(10,10))
nx.draw_networkx(g,
                 pos = node_positions, 
                 width = width,
                 arrows = False,
                 node_size = 0,
                 with_labels = False,
                 node_color = colors)

# plt.ylim(2640000,2660000)
# plt.xlim(710000,750000)

plt.plot([],[],'k-', lw = np.array(width).min(), label = 'Q_min =' + str(np.floor(np.array(width).min() * 150)))
plt.plot([],[],'k-', lw = np.array(width).max(), label = 'Q_max =' + str(np.array(width).max() * 150))
plt.plot([],[],'w', label=' ')

plt.plot(source_x, source_y, 'X', mfc = 'orange', mec = 'w', ms = 20, label = 'sum Q_in (orange) =' + str(sum([g.nodes[n]['discharge'] for n in sources])))

plt.plot(sink_x, sink_y, 'm*', mec = 'w', ms = 30, label = 'sum Q_out (purple) =' + str(sum([g.nodes[n]['discharge'] for n in sinks])))

plt.legend(fontsize='xx-large')

plt.savefig('simplified_network.png', dpi = 150)
plt.close()
