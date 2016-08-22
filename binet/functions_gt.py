from graph_tool.all import Graph,minimize_blockmodel_dl
from graph_tool.draw import sfdp_layout
#Note: I use a separate file because of conflicts between networkx and graph_tool

def get_pos(net_,s=None,t=None,node_id = None,comms = False):
    '''Returns the sfdp layout of the given network (https://graph-tool.skewed.de/static/doc/draw.html#graph_tool.draw.sfdp_layout).
    If comms is True it also returns a community id for each node.    
    WARNING: Requires graph_tool
    '''
    node_id = 'node_id' if node_id is None else node_id
    s = net_.columns.values[0] if s is None else s
    t = net_.columns.values[1] if t is None else t
    net = net_[[s,t]]
    
    net_norm = pd.DataFrame(list(set(net[s].values)|set(inds[t].values)),columns=[node_id]).reset_index()
    net = pd.merge(net,net_norm.rename(columns={node_id:s,'index':'index_'+s}),how='left')
    net = pd.merge(net,net_norm.rename(columns={node_id:t,'index':'index_'+t}),how='left')

    G = Graph(directed=False)
    G.add_edge_list(zip(net['index_'+s].values,net['index_'+t].values))
    #pos = graph_draw(G)
    pos = sfdp_layout(G)
    out = pd.merge(net_norm,pd.DataFrame([[i]+list(pos[i]) for i in net_norm['index']] ,columns=['index','x','y'])).drop('index',1)
    if comms:
        state = minimize_blockmodel_dl(G)
        blocks = state.get_blocks()
        out = pd.merge(out,pd.merge(net_norm,pd.DataFrame([(i,blocks[i]) for i in net_norm['index']],columns=['index','c'])).drop('index',1))
    return out
    