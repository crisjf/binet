try:
    from graph_tool.all import Graph,minimize_blockmodel_dl
    from graph_tool.draw import sfdp_layout
except:
    print 'Warning: No module named graph_tool found.'
from pandas import DataFrame,merge

def get_pos(net_,s=None,t=None,node_id = None,comms = False,progress=False,C=None):
    '''Returns the sfdp layout of the given network (https://graph-tool.skewed.de/static/doc/draw.html#graph_tool.draw.sfdp_layout).
    If comms is True it also returns a community id for each node.    
    WARNING: Requires graph_tool

    Parameters
    ----------
    net_ : DataFrame
        DataFrame with a list of edges.
    s : str (optional)
        Column label of the source node.
        If not provided it will be set as the first column of net_.
    t : str (optional)
        Column label of the target node.
        If not provided it will be set as the second column of net_.
    node_id : str (optional)
        Label to use for the node_id columns in the output.
    comms : boolean (False)
        If True it will detect communities as well.
    progress : boolean (False)
        If True it will print the progress. 
        Used for debugging.
    C : float (optional, default: 0.2)
        Relative strength of repulsive forces.
        See graph_tool.draw.sfdp_layout()
    
    '''
    node_id = 'node_id' if node_id is None else node_id
    s = net_.columns.values[0] if s is None else s
    t = net_.columns.values[1] if t is None else t
    net = net_[[s,t]]
    
    net_norm = DataFrame(list(set(net[s].values)|set(net[t].values)),columns=[node_id]).reset_index()
    net = merge(net,net_norm.rename(columns={node_id:s,'index':'index_'+s}),how='left')
    net = merge(net,net_norm.rename(columns={node_id:t,'index':'index_'+t}),how='left')

    G = Graph(directed=False)
    G.add_edge_list(zip(net['index_'+s].values,net['index_'+t].values))
    if progress:
        print 'Calculating positions'
    if C is not None:
        pos = sfdp_layout(G,C=C)
    else:
        pos = sfdp_layout(G)
    out = merge(net_norm,DataFrame([[i]+list(pos[i]) for i in net_norm['index']] ,columns=['index','x','y'])).drop('index',1)
    if comms:
        if progress:
            print 'Calculating communities'
        state = minimize_blockmodel_dl(G)
        blocks = state.get_blocks()
        out = merge(out,merge(net_norm,DataFrame([(i,blocks[i]) for i in net_norm['index']],columns=['index','c'])).drop('index',1))
    return out



def get_comms(net_,s=None,t=None,node_id = None):
    '''
    Returns a community id for each node.    
    WARNING: Requires graph_tool

    Parameters
    ----------
    net_ : DataFrame
        DataFrame with a list of edges.
    s : str (optional)
        Column label of the source node.
        If not provided it will be set as the first column of net_.
    t : str (optional)
        Column label of the target node.
        If not provided it will be set as the second column of net_.
    node_id : str (optional)
        Label to use for the node_id columns in the output.
    
    '''
    node_id = 'node_id' if node_id is None else node_id
    s = net_.columns.values[0] if s is None else s
    t = net_.columns.values[1] if t is None else t
    net = net_[[s,t]]
    net_norm = DataFrame(list(set(net[s].values)|set(net[t].values)),columns=[node_id]).reset_index()
    net = merge(net,net_norm.rename(columns={node_id:s,'index':'index_'+s}),how='left')
    net = merge(net,net_norm.rename(columns={node_id:t,'index':'index_'+t}),how='left')
    G = Graph(directed=False)
    G.add_edge_list(zip(net['index_'+s].values,net['index_'+t].values))
    state = minimize_blockmodel_dl(G)
    blocks = state.get_blocks()
    out = merge(net_norm,merge(net_norm,DataFrame([(i,blocks[i]) for i in net_norm['index']],columns=['index','c'])).drop('index',1))
    return out
