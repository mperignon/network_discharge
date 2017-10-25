import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial

def do_kdtree(combined_x_y_arrays,points):
    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points)
    return indexes


edges = pd.read_csv('raw_edges.csv')
file_nodes = pd.read_csv('raw_nodes.csv')



from_nodes = np.array(edges[['startpt_x', 'startpt_y']])
nodes = np.vstack((from_nodes, np.array(edges[['endpt_x', 'endpt_y']]))).astype(np.int)

unique_nodes = np.unique(np.sort(nodes, axis=1), axis=0).astype(np.int)
unique_nodes.view('float64,float64').sort(order=['f1'], axis=0)



pts = do_kdtree(unique_nodes, np.sort(edges[['startpt_x','startpt_y']], axis=1))
edges['START_pt'] = pts

pts = do_kdtree(unique_nodes, np.sort(edges[['endpt_x','endpt_y']], axis=1))
edges['END_pt'] = pts

edges2 = edges[edges['START_pt'] != edges['END_pt']]

edgesdf = edges2[['START_pt', 'END_pt', 'meanWidth', 'length']]
edgesdf.columns = ['START_pt', 'END_pt', 'Width', 'length']
edgesdf.to_csv('simple_alledges.csv')




nodes = np.hstack((np.zeros((len(unique_nodes),2)), unique_nodes))
nodes[:,0] = np.arange(len(unique_nodes))

pts = do_kdtree(file_nodes[['X','Y']], unique_nodes)
nodes[:,1] = np.array(file_nodes['cat'][pts])

nodedf = pd.DataFrame(data = nodes, columns=['NodeID', 'cat', 'x', 'y'])

nodedf.to_csv('simple_nodes_xy.csv')

