from networkx import Graph,minimum_spanning_tree,number_connected_components,is_connected,connected_component_subgraphs
from pandas import DataFrame,merge,concat,read_csv
from numpy import array,matrix,mean,std,log
from scipy.interpolate import interp1d
from copy import deepcopy
import json

def getJenksBreaks( dataList, numClass ):
    dataList.sort()
    mat1 = []
    for i in range(0,len(dataList)+1):
        temp = []
        for j in range(0,numClass+1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(0,len(dataList)+1):
        temp = []
        for j in range(0,numClass+1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1,numClass+1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2,len(dataList)+1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2,len(dataList)+1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1,l+1):
            i3 = l - m + 1
            val = float(dataList[i3-1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2,numClass+1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(dataList)
    kclass = []
    for i in range(0,numClass+1):
        kclass.append(0)
    kclass[numClass] = float(dataList[len(dataList) - 1])
    countNum = numClass
    while countNum >= 2:#print "rank = " + str(mat1[k][countNum])
        id = int((mat1[k][countNum]) - 2)
        #print "val = " + str(dataList[id])
        kclass[countNum - 1] = dataList[id]
        k = int((mat1[k][countNum] - 1))
        countNum -= 1
    return kclass
    


def CalculateComplexity(M,th=0.0001):
    '''Calculates the Economic Complexity Index following the method of reflections presented in
    Hidalgo and Hausmann 2010, The building blocks of economic complexity.
    THIS FUNCTION NEEDS TO BE TESTED!!!

    Parameters
    ----------
    M : numpy array with only zeros and ones.
        Represents the adjacency matrix of the bipartite network. Both the sum of the rows and the sum of the columns must contain non-zero entries.
    th : scalar.
        The stopping criteria for the method of reflections.
    
    Returns
    ----------
    ECI : list
        Economic Complexity Index for the rows of M.
    PCI : list
        Product Compleixty Index for the columns of M.
    '''
    kc0 = M.sum(1)
    kp0 = M.sum(0)
    if any(kc0==0)|any(kp0==0):
        raise NameError('One dimension has a null sum')
    kcinv = 1./kc0
    kpinv = 1./kp0
    it_count=0
    while True:
        it_count+=1
        kc1 = kcinv*array((matrix(M)*matrix(kp0).T).T)[0]
        kp1 = kpinv*array((matrix(M.T)*matrix(kc0).T).T)[0]
        kc2 = kcinv*array((matrix(M)*matrix(kp1).T).T)[0]
        kp2 = kpinv*array((matrix(M.T)*matrix(kc1).T).T)[0]
        if all(abs(kc2-kc0)<th) & all(abs(kp2-kp0)<th):
            kc = kc2[:]
            kp = kp2[:]
            break
        kc0 = kc2[:]
        kp0 = kp2[:]
        if it_count>=1000:
            raise NameError('Method of reflections did not converge after 1000 iterations')
    ECI = (kc-mean(kc))/std(kc)
    PCI = (kp-mean(kp))/std(kp)
    return ECI.tolist(),PCI.tolist()


def order_columns(net,s=None,t=None):
    '''
    Makes sure that the label for the source node is larger than the label for the target node.
    It flips the nodes if t>s.

    Parameters
    ----------
    net : pandas.DataFrame
        List of links
    s,t : str (optional)
        Label of the column for source and target nodes.

    Returns
    -------
    net : pandas.DataFrame
        List of links.
    '''
    s = net.columns.values[0] if s is None else s
    t = net.columns.values[1] if t is None else t
    B = net[s]>net[t]
    if B.sum() !=0:
        S,T = zip(*net[B][[s,t]].values)
        net.loc[B,s]=T
        net.loc[B,t]=S
    return net



def flatten(dis,s=None,t=None,agg_method='sum'):
    """
    Orders the nodes such that the source codes are always smaller than the target codes, and aggregates.
    It is used to flatten a directed network in a table.

    Parameters
    ----------
    dis : pandas.DataFrame
        Table with source,target plus other columns that will be aggregated
    s,t : str (optional)
        Labels on dis to use as source and target
    agg_method : str (default='sum')
        Aggregation method
        Can be 'sum' or 'max'

    Returns
    -------
    new : pandas.DataFrame

    """
    s = dis.columns.values[0] if s is None else s
    t = dis.columns.values[1] if t is None else t
    new = order_columns(deepcopy(dis),s=s,t=t)
    if len(new.columns) >2:
        if agg_method=='sum':
            new = new.groupby([s,t]).sum().reset_index()
        elif agg_method=='max':
            new = new.groupby([s,t]).max().reset_index()
        else:
            raise NameError('Unrecognized aggregation method '+agg_method)
    else:
        new = new.drop_duplicates()
    return new


def build_connected(net,th,s=None,t=None,w=None,directed=False,progress=True):
    """Builds a connected network out of a set of weighted edges and a threshold.

    Parameters
    ----------
    net : pandas DataFrame
            Contains at least three columns: source, target, and weight
    th : float
            Threshold to use for the network.
    s,t,w : (optional) strings
            Names of the columns in net to use as source, target, and weight, respectively.
            If it is not provided, the first three columns as assumed to be s,t,and w, respectively.
    directed : False (default)
            Wether to consider the network as directed or as undirected.

    Returns
    -------
    edges : pandas DataFrame
        Contains the links of the resulting network. The weights are also returned.

    Examples
    --------
    >>> 

    See Also
    --------
    mcp, tnet

    Notes
    -----
    Set the threshold low to begin with.
    The best threshold is such that the average degree of the network is close to 3.
    """
    s = net.columns.values[0] if s is None else s
    t = net.columns.values[1] if t is None else t
    w = net.columns.values[2] if w is None else w

    dis = net[[s,t,w]]
    dis[s] = dis[s].astype(int).astype(str)
    dis[t] = dis[t].astype(int).astype(str)

    G = Graph()
    G.add_edges_from(list(set([tuple(set(edge)) for edge in zip(dis[s],dis[t])])))
    if not is_connected(G):
        raise NameError('The provided network is not connected.')

    if not directed:
        G = Graph()    
        G.add_edges_from(zip(dis[s].values,dis[t].values,[{'weight':-f} for f in dis[w]]))
        T = minimum_spanning_tree(G)
        T.add_edges_from([(u,v,{'weight':-we}) for u,v,we in dis[dis[w]>=th].values.tolist()])
        if progress:
            print 'N edges:',len(T.edges())
            print 'N nodes:',len(T.nodes())
        out = []
        for u,v in T.edges():
            out.append((u,v,-T.get_edge_data(u, v)['weight']))
        edges = DataFrame(out,columns=[s,t,w])
        edges[s] = edges[s].astype(net.dtypes[s])
        edges[t] = edges[t].astype(net.dtypes[t])
        return edges
    else:
        new = order_nodes(dis,s=None,t=None)
        G = Graph()
        G.add_edges_from(zip(new[s].values,new[t].values,[{'weight':-f} for f in new[w]]))
        T = minimum_spanning_tree(G)
        Tedges = concat([DataFrame(T.edges(),columns=[s,t]).reset_index(),DataFrame(T.edges(),columns=[t,s]).reset_index()])
        Tedges = merge(Tedges,dis).dropna()
        idx = Tedges.groupby(['index'])[w].transform(max) == Tedges[w]
        edges_out = Tedges[idx][[s,t]]
        edges_out = merge(concat([dis[dis[w]>=th][[s,t]],edges_out]).drop_duplicates(),dis)
        edges_out[s] = edges_out[s].astype(net.dtypes[s])
        edges_out[t] = edges_out[t].astype(net.dtypes[t])
        return edges_out
    

def calculateRCA(data,c='',p='',x='',shares=False):
    '''
    Returns the RCA expressed in data

    Parameters
    ----------
    data : pandas.DataFrame
        Raw data. It has source,target,volume (trade, number of people etc.).
    c,p,x : str (optional)
        Labels of the columns in data used for source,target,volume
    shares : boolean (False)
        If True it will also return the shares used to calculate the RCA

    Returns
    -------
    RCA : pandas.DataFrame
        Table with the RCAs, with the columns c,p,x,RCA
        If shares is True it also includes:
            s_c : Share of X_cp over X_c
            s_p : Share of X_cp over X_p
    '''
    c = data.columns.values[0] if c == '' else c
    p = data.columns.values[1] if p == '' else p
    x = data.columns.values[2] if x == '' else x
    data_ = data[[c,p,x]]
    data_ = merge(data_,data_.groupby(c).sum()[[x]].rename(columns={x:x+'_'+c}).reset_index()
                         ,how='inner',left_on=c,right_on=c)
    data_ = merge(data_,data_.groupby(p).sum()[[x]].rename(columns={x:x+'_'+p}).reset_index()
                         ,how='inner',left_on=p,right_on=p)
    X = float(data_.sum()[x])
    data_['RCA'] = (data_[x].astype(float)/data_[x+'_'+p].astype(float))/(data_[x+'_'+c].astype(float)/X)
    if shares:
        data_['s_'+c] = (data_[x].astype(float)/data_[x+'_'+c].astype(float)) 
        data_['s_'+p] = (data_[x].astype(float)/data_[x+'_'+p].astype(float))
        return data_[[c,p,x,'RCA','s_'+c,'s_'+p]]
    return data_[[c,p,x,'RCA']]


def read_cyto(cyto_file,node_id='shared_name'):
    """Reads the positions of the nodes in cytoscape"""
    with open(cyto_file) as data_file:
        cyto = json.load(data_file)
    out = []
    for node in cyto['elements']['nodes']:
        out.append( (node['data']['shared_name'],node['position']['x'],node['position']['y']))
    return DataFrame(out,columns=[node_id,'x','y'])


def df_interp(df,by=None,x=None,y=None,kind='linear'):
    '''Groups df by the given column, and interpolates the missing values.
    THIS ONE CAN ALSO BE PARALELLIZED'''
    by = df.columns.values[0] if by is None else by
    x  = df.columns.values[1] if x is None else x
    y  = df.columns.values[2] if y is None else y
    interp = []
    cs = list(set(df[by].values))
    for c in cs:
        gc = df[df[by]==c].sort_values(by=x)
        X = gc[x]
        Y = gc[y]
        f = interp1d(X, Y, kind=kind)
        xx = array(range(min(X),max(X)+1))
        yy = f(xx)
        interp+=zip(len(xx)*[c],xx,yy)
    return DataFrame(interp,columns=[by,x,y])





def build_html(nodes,edges,node_id =None,source_id =None,target_id=None,size_id=None,weight_id = None,x=None,y=None,color=None,props = None,progress=True,max_colors=15):
    """
    Creates an html file with a d3plus visualization of the network from the dataframes nodes and edges.
    NEEDS A FUNCTION TO HANDLE WITH MANY DIFFERENT COLORS
    """

    node_id = nodes.columns.values[0] if node_id is None else node_id
    source_id = edges.columns.values[0] if source_id is None else source_id
    target_id = edges.columns.values[1] if target_id is None else target_id
    props = [] if props is None else props
    x = 'x' if x is None else x
    y = 'y' if y is None else y
    
    if progress:
        print 'node_id  :',node_id
        print 'source_id:',source_id
        print 'target_id:',target_id
        print 'size_id  :',size_id
        print 'weight_id:',weight_id
        print 'x,y      :',x,',',y
        print 'color    :',color
        print 'props    :',props

    if color is not None:
        if len(set(nodes[color].values))>max_colors:
            raise NameError('Too many different colors, try using Jenks Natural Breaks: '+str(len(set(nodes[color].values))))

    sample_data = '[\n'
    positions = '[\n'
    connections = '[\n'
    
    for entry in nodes.itertuples():
        row = dict(zip(['index']+nodes.columns.values.tolist(),entry))
        sd = ['"n_id":'+'"'+str(row[node_id])+'"']
        if size_id is not None:
            ad.append('"size":'+str(log(row[size_id])))
        
        if color is not None:
            sd.append('"color":'+'"'+str(row[color])+'"')
        for prop in props:
            sd.append('"'+str(prop)+'":'+'"'+unicode(row[prop])+'"')
        sample_data+= '{'+','.join(sd)+'},\n'
        positions += '{"n_id":"'+str(row[node_id]) +'", "x":'+str(row[x])+',"y": '+str(row[y])+'},\n'

    for index,row in edges.iterrows():
        cn = ['"source":'+'"'+str(int(row[source_id]))+'"',
              '"target":'+'"'+str(int(row[target_id]))+'"']
        if weight_id is not None:
            cn.append('"strength":'+str(row[weight_id]))
        connections += '{'+','.join(cn)+'},\n'

    sample_data = sample_data[:-2]+'\n]\n'
    positions = positions[:-2]+'\n]\n'
    connections = connections[:-2]+'\n]\n'
    
    viz = """var visualization = d3plus.viz()
        .container("#viz")
        .type("network")
        .data(sample_data)
        .nodes(positions)
        .id("n_id")
        """
    if weight_id is not None:
        viz += '.edges({"value": connections,"size": "strength"})\n'
    else:
        viz += '.edges({"value": connections})\n'
        
    if size_id is not None:
        viz += '.size("size")\n'
    
    ttip = ['"n_id"']+['"'+str(p)+'"' for p in props]
    if color is not None:
        viz +='\t\t.color("color")\n'
        ttip.append('"color"')
    if size_id is not None:
        ttip.append('"size"')
    viz += '\t\t.tooltip('+'['+','.join(ttip)+']'+')\n'
    viz += "\t\t.draw()"

    html = """<!doctype html>
    <meta charset="utf-8">
    <script src="http://www.d3plus.org/js/d3.js"></script>
    <script src="http://www.d3plus.org/js/d3plus.js"></script>
    <div id="viz"></div>
    <script>
    // create sample dataset
    var sample_data = """ 
    html += sample_data + 'var positions =' + positions + 'var connections ='+connections
    html += viz +'\n</script>'
    return html


def compare_nets(M1_,M2_):
    '''
    Given two BiGraph objects it compares them on different dimensions.
    (NEEDS TO BE TESTED)
    '''
    n1 = 'Unnamed' if M1_.name == '' else M1_.name
    n2 = 'Unnamed' if M2_.name == '' else M2_.name

    M1 = deepcopy(M1_)
    M2 = deepcopy(M2_)
    if M1.net is None:
        M1.build_net(progress=False)
    if M2.net is None:
        M2.build_net(progress=False)
    edges1 = M1.net[[M1.c,M1.p]]
    edges2 = M2.net[[M2.c,M2.p]]

    nodes1 = set(M1.nodes(M1.c)[M1.c].values)
    nodes2 = set(M2.nodes(M2.c)[M2.c].values)
    c_total_nodes     = len(nodes1|nodes2)
    c_repeated_nodes  = len(nodes1.intersection(nodes2))
    c_f_missing_nodes = 1.-len(nodes1.intersection(nodes2))/float(len(nodes1|nodes2))
    c_f_missing_1     = 1.-len(nodes1)/float(len(nodes1|nodes2))
    c_f_missing_2     = 1.-len(nodes2)/float(len(nodes1|nodes2))

    nodes_c = list(nodes1.intersection(nodes2))
    nodes = DataFrame(nodes_c,columns=['node_id'])
    edges1 = merge(edges1,nodes,how='right',left_on=M1.c,right_on='node_id').drop('node_id',1)
    edges2 = merge(edges2,nodes,how='right',left_on=M2.c,right_on='node_id').drop('node_id',1)

    nodes1 = set(M1.nodes(M1.p)[M1.p].values)
    nodes2 = set(M2.nodes(M2.p)[M2.p].values)
    p_total_nodes     = len(nodes1|nodes2)
    p_repeated_nodes  = len(nodes1.intersection(nodes2))
    p_f_missing_nodes = 1.-len(nodes1.intersection(nodes2))/float(len(nodes1|nodes2))
    p_f_missing_1     = 1.-len(nodes1)/float(len(nodes1|nodes2))
    p_f_missing_2     = 1.-len(nodes2)/float(len(nodes1|nodes2))        

    nodes_p = list(nodes1.intersection(nodes2))
    nodes = DataFrame(nodes_p,columns=['node_id'])
    edges1 = merge(edges1,nodes,how='right',left_on=M1.p,right_on='node_id').drop('node_id',1)
    edges2 = merge(edges2,nodes,how='right',left_on=M2.p,right_on='node_id').drop('node_id',1)
    
    edges1['edge'] = edges1[M1.c].astype(str)+'-'+edges1[M1.p].astype(str)
    edges2['edge'] = edges2[M2.c].astype(str)+'-'+edges2[M1.p].astype(str)
    edges1 = set(edges1['edge'].values.tolist())
    edges2 = set(edges2['edge'].values.tolist())

    e_total_edges     = len(edges1|edges2)
    e_f_missing_edges = 1.-len(edges1.intersection(edges1))/float(len(edges1|edges2))
    e_f_missing_1     = 1.-len(edges1)/float(len(edges1|edges2))
    e_f_missing_2     = 1.-len(edges2)/float(len(edges1|edges2))

    M1.filter_nodes(nodes_c=nodes_c,nodes_p=nodes_p)
    M2.filter_nodes(nodes_c=nodes_c,nodes_p=nodes_p)
    M1.build_net(progress=False)
    M2.build_net(progress=False)

    edges1 = M1.net[[M1.c,M1.p]]
    edges2 = M2.net[[M2.c,M2.p]]
    edges1['edge'] = edges1[M1.c].astype(str)+'-'+edges1[M1.p].astype(str)
    edges2['edge'] = edges2[M2.c].astype(str)+'-'+edges2[M1.p].astype(str)
    edges1 = set(edges1['edge'].values.tolist())
    edges2 = set(edges2['edge'].values.tolist())

    f_total_edges     = len(edges1|edges2)
    f_f_missing_edges = 1.-len(edges1.intersection(edges1))/float(len(edges1|edges2))
    f_f_missing_1     = 1.-len(edges1)/float(len(edges1|edges2))
    f_f_missing_2     = 1.-len(edges2)/float(len(edges1|edges2))

    cp = merge(M1.projection(M1.c).rename(columns={'fi':'fi_1'}),M2.projection(M2.c).rename(columns={'fi':'fi_2'}),how='inner')
    proj_c = corrcoef(cp['fi_1'],cp['fi_2'])[0,1]
    cp = merge(M1.projection(M1.p).rename(columns={'fi':'fi_1'}),M2.projection(M2.p).rename(columns={'fi':'fi_2'}),how='inner')
    proj_p = corrcoef(cp['fi_1'],cp['fi_2'])[0,1]

    c_m1 = M1.c
    c_m2 = M2.c
    p_m1 = M1.p
    p_m2 = M2.p

    print 'Summary of comparing networks '+n1+' with '+n2
    print ','.join(list(set([c_m1,c_m2])))
    print '\tTotal nodes          :',c_total_nodes
    print '\tTotal repeated nodes :',c_repeated_nodes
    print '\tFracion missing nodes:',c_f_missing_nodes
    print '\tMissing nodes on 1   :',c_f_missing_1
    print '\tMissing nodes on 2   :',c_f_missing_2
    print ''
    print ','.join(list(set([p_m1,p_m2])))
    print '\tTotal nodes          :',p_total_nodes
    print '\tTotal repeated nodes :',p_repeated_nodes
    print '\tFracion missing nodes:',p_f_missing_nodes
    print '\tMissing nodes on 1   :',p_f_missing_1
    print '\tMissing nodes on 2   :',p_f_missing_2
    print ''
    print ','.join(list(set([str(c_m1)+'-'+str(p_m1),str(c_m2)+'-'+str(p_m2)])))
    print '\tTotal edges          :',e_total_edges
    print '\tFracion missing edges:',e_f_missing_edges
    print '\tMissing edges on 1   :',e_f_missing_1
    print '\tMissing edges on 2   :',e_f_missing_2
    print ''
    print 'Filtered '+','.join(list(set([str(c_m1)+'-'+str(p_m1),str(c_m2)+'-'+str(p_m2)])))
    print '\tTotal edges          :',f_total_edges
    print '\tFracion missing edges:',f_f_missing_edges
    print '\tMissing edges on 1   :',f_f_missing_1
    print '\tMissing edges on 2   :',f_f_missing_2
    print ''
    print 'Projection ',','.join(list(set([c_m1,c_m2])))
    print 'Correlation between fis:',proj_c
    print ''
    print 'Projection ',','.join(list(set([p_m1,p_m2])))
    print 'Correlation between fis:',proj_p
    print ''
