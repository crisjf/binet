from networkx import Graph,minimum_spanning_tree,number_connected_components,is_connected,connected_component_subgraphs
from pandas import DataFrame,merge,concat,read_csv
from numpy import array,matrix,mean,std,log,sqrt,exp,log10,array_split
from scipy.interpolate import interp1d
from sklearn import neighbors
from copy import deepcopy
try:
    from community import best_partition
except:
    print 'Warning: No module named community found.'
import json


def communities(net,s=None,t=None,node_id=None):
    '''
    Calculates the best partition from a given set of edges.

    Parameters
    ----------
    net : pandas.DataFrame
        Has the source and the target.
    s,t : str (optional)
        Source and target columns.
        If not provided it will take the first and the second.
    node_id : str (optional)
        Name for the column that identifies the nodes.
        If not provided, the column will be names 'node_id'.

    Returns
    -------
    partition : pandas.DataFrame
        Table with two columns, node_id and community_id
    '''
    s = net.columns.values[0] if s == None else s
    t = net.columns.values[1] if t == None else t
    node_id = 'node_id' if node_id is None else node_id
    G = Graph()
    G.add_edges_from(net[[s,t]].values)
    part = best_partition(G)
    part = DataFrame(part.items(),columns=[node_id,'community_id'])
    print 'Number of communities:',len(set(part['community_id']))
    return part



def allCombinations(data,c=None,p=None,t=None,bipartite=True):
    '''
    Creates all possible combinations between t,c,p.

    If t is not provided, it will only create combinations between c and p

    Parameters
    ----------
    data : pandas.DataFrame
        Table with c and p as columns.
    c,p : str
        Columns to use as nodes.
    t : str
        Group. Usually the time column.
    bipartite : boolean (True)
        If True it will treat the network as bipartite
        If False it will treat it as monopartite.

    Returns
    -------
    combinations : pandas.DataFrame
        DataFrame with all the possible combinations.
        If t is not provided, it has two columns with c and p.
        If t is provided, it has three columns, with t, c and p.
    '''
    combs = []
    if t is None:
        c = data.columns.values[0] if c is None else c
        p = data.columns.values[1] if p is None else p
        if bipartite:
            cs = set(data[c])
            ps = set(data[p])
            for cc in cs:
                for pp in ps:
                    combs.append((cc,pp))
        else:
            cs = set(data[c])|set(data[p])
            for cc in cs:
                for pp in cs:
                    if cs!=ps:
                        combs.append((cc,pp))
        combs = DataFrame(combs,columns=[c,p])
    else:
        c = data.columns.values[1] if c is None else c
        p = data.columns.values[2] if p is None else p
        for y in set(data[t]):
            datay = data[data[t]==y]
            if bipartite:
                cs = set(datay[c])
                ps = set(datay[p])
                for cc in cs:
                    for pp in ps:
                        combs.append((y,cc,pp))
            else:
                cs = set(datay[c])|set(datay[p])
                for cc in cs:
                    for pp in cs:
                        if cc!=pp:
                            combs.append((y,cc,pp))
        combs = DataFrame(combs,columns=[t,c,p])
    return combs


def countBoth(data,t=None,c=None,p=None,add_marginal=True):
    '''
    Counts the coocurrences of pairs of ps on cs. 
    For example, it counts the coocurrences of products in countries.
    If year if provided, it will do it yearly.

    Paramters
    ---------
    data : pandas.DataFrame
        Data of a network connecting c and p. 
        Can also be over time.
    t,c,p : str
        Name of the columns to use as t, c, and p.
    add_marginal : boolean (True)
        If True it will also add the sum of occurrences of p.
    '''
    if t is not None:
        c = data.columns.values[1] if c is None else c
        p = data.columns.values[2] if p is None else p
        data_ = data[[t,c,p]].drop_duplicates()
    else:
        c = data.columns.values[0] if c is None else c
        p = data.columns.values[1] if p is None else p
        data_ = data[[c,p]].drop_duplicates()
    net = merge(data_,data_.rename(columns={p:p+'p'}))
    if t is not None:
        net = net.groupby([t,p,p+'p']).count()[[c]].reset_index().rename(columns={c:'N_both'})
    else:
        net = net.groupby([p,p+'p']).count()[[c]].reset_index().rename(columns={c:'N_both'})
    net = net[net[p]!=net[p+'p']]
    if add_marginal:
        if t is not None:
            data_ = data_.groupby([t,p]).count()[[c]].reset_index().rename(columns={c:'N_'+p})
        else:
            data_ = data_.groupby([p]).count()[[c]].reset_index().rename(columns={c:'N_'+p})
        net = merge(net,data_)
        net = merge(net,data_.rename(columns={p:p+'p','N_'+p:'N_'+p+'p'}))
    return net



import statsmodels.api as sm
def residualNet(data,s=None,t=None,x=None,g=None,numericalControls=[],categoricalControls=[],addDummies=True):
    '''
    Given the data on a network of the form source,target,flow, it controls for the given variables, and takes the residual.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw data. It has source,target,volume (trade, number of people etc.).
    s,t,x : str (optional)
        Labels of the columns in data used for source,target,volume. 
        If not provided it will use the first, second, and third.
    g : str (optional)
        If provided, it will run independent regressions for each value of the g column.
    numericalControls : list
        List of columns to use as numerical controls.
    categoricalControls : list
        List of columns to use as categorical controls.
    addDummies : boolean (True)
        If True it will add controls for each node.
        
    Returns
    -------
    net : pandas.Dataframe
        Table with g,s,t,x,x_res, where x_res is the residual of regressing x on the given control variables.
    '''
    s = data.columns.values[0] if s is None else s
    t = data.columns.values[1] if t is None else t
    x = data.columns.values[2] if x is None else x

    if g is None:
        data_ = data[[s,t,x]+numericalControls+categoricalControls]
    else:
        data_ = data[[g,s,t,x]+numericalControls+categoricalControls]

    _categoricalControls = []
    for var in set(categoricalControls):
        vals = list(set(data_[var]))
        for v in vals[1:]:
            _categoricalControls.append(var+'_'+str(v))
            data_[var+'_'+str(v)]=0
            data_.loc[data_[var]==v,var+'_'+str(v)]=1

    if addDummies:
        nodes = list(set(data[s])|set(data[t]))
        for node in nodes[1:]:
            _categoricalControls.append('node_'+str(node))
            data_['node_'+str(node)]=0
            data_.loc[(data_[s]==node)|(data_[t]==node),'node_'+str(node)]=1

    if g is not None:
        out = []
        for gg in set(data_[g]):
            data_g = data_[data_[g]==gg]
            Y = data_g[x].values
            X = data_g[list(set(numericalControls))+list(set(_categoricalControls))].values
            X = sm.add_constant(X)
            try:
                model = sm.OLS(Y,X).fit()
                data_g[x+'_res'] = Y-model.predict(X)
            except:
                print 'Not able to fit group ',g
                data_g[x+'_res'] = Y
            data_g[g] = gg
            out.append(data_g[[g,s,t,x,x+'_res']])
        data_ = concat(out)[[g,s,t,x,x+'_res']]
    else:
        Y = data_[x].values
        X = data_[list(set(numericalControls))+list(set(_categoricalControls))].values
        X = sm.add_constant(X)
        model = sm.OLS(Y,X).fit()
        data_[x+'_res'] = Y-model.predict(X)
        data_ = data_[[s,t,x,x+'_res']]
    print 'R2 for',x,'with dummies' if addDummies else 'without dummies',model.rsquared
    return data_



def _residualNet(data,uselog=True,c=None,p=None,x=None,useaggregate=True,numericalControls=[],categoricalControls=[]):
    '''
    Given the data on a bipartite network of the form source,target,flow

    Parameters
    ----------
    data : pandas.DataFrame
        Raw data. It has source,target,volume (trade, number of people etc.).
    c,p,x : str (optional)
        Labels of the columns in data used for source,target,volume. 
        If not provided it will use the first, second, and third.
    numericalControls : list
        List of columns to use as numerical controls.
    categoricalControls : list
        List of columns to use as categorical controls.
    uselog : boolean (True)
        If True it will use the logarithm of the provided weight.
    useaggregate : boolean (True)
        If true it will calculate the aggregate of the volume on both sides (c and p) and use as numbercal controls.
        
    Returns
    -------
    net : pandas.Dataframe
        Table with c,p,x,x_res, where x_res is the residual of regressing x on the given control variables.
    '''
    c = data.columns.values[0] if c is None else c
    p = data.columns.values[1] if p is None else p
    x = data.columns.values[2] if x is None else x
    data_ = data[[c,p,x]+numericalControls+categoricalControls]
    if useaggregate:
        data_ = merge(data_,data.groupby(c).sum()[[x]].reset_index().rename(columns={x:x+'_'+c}))
        data_ = merge(data_,data.groupby(p).sum()[[x]].reset_index().rename(columns={x:x+'_'+p}))
        numericalControls+=[x+'_'+c,x+'_'+p]
    if uselog:
        data_ = data_[data_[x]!=0]
        data_[x] = log10(data_[x])
        if useaggregate:
            data_[x+'_'+c] = log10(data_[x+'_'+c])
            data_[x+'_'+p] = log10(data_[x+'_'+p])
    _categoricalControls = []
    for var in ser(categoricalControls):
        vals = list(set(data_[var]))
        for v in vals[1:]:
            _categoricalControls.append(var+'_'+str(v))
            data_[var+'_'+str(v)]=0
            data_.loc[data_[var]==v,var+'_'+str(v)]=1

    Y = data_[x].values
    X = data_[list(set(numericalControls))+list(set(_categoricalControls))].values
    X = sm.add_constant(X)

    model = sm.OLS(Y,X).fit()
    data_[x+'_res'] = Y-model.predict(X)
    return data_[[c,p,x,x+'_res']]


def densities(m,fi,normalize=True):
    '''
    Calculates the density of related entities.
    
    Parameters
    ----------
    m : pandas.DataFrame
        Groups to which group the entities.
        For example, they can be countries and products produced by countries.
        Has the colums:
            group_id,entity_id
    fi : pandas.DataFrame
        Table with relatedness between entities.
        Has the columns:
            entity_id_source,entity_id_target,weight
    normalize : boolean (True)
        If False, it will not normalize by the degree.
    
    Returns
    -------
    W : pandas.DataFrame
        Dataframe with the densities.
        Has the columns:
            group_id,entity_id,density,mcp
    '''
    g,side = m.columns.values[:2].tolist()
    s,t,w = fi.columns.values[:3].tolist()
    fi = concat([fi,fi[[s,t,w]].rename(columns={s:t,t:s})]).drop_duplicates().rename(columns={s:side})
    W = merge(m.rename(columns={side:t}),fi,how='left').fillna(0).groupby([g,side]).sum()[[w]].reset_index().rename(columns={w:'num'})
    fi = fi.groupby(side).sum()[[w]].reset_index().rename(columns={w:'den'})
    W = merge(W,fi)
    if normalize:
        W['w'] = W['num']/W['den']
    else:
        W['w'] = W['num']
    W = W[[g,side,'w']]
    m['mcp'] = 1
    m = merge(m,DataFrame([(gg,ss) for gg in set(m[g]) for ss in set(m[side])],columns=[g,side]),how='outer').fillna(0)
    W = merge(W,m,how='outer').fillna(0)
    return W

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
    '''
    Calculates the Economic Complexity Index following the method of reflections presented in
    Hidalgo and Hausmann 2010, The building blocks of economic complexity.

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


def CalculateComplexityPlus(X,th=0.001,max_iter=5000):
    '''
    Calculates the Economic Complexity Index Plus .
    X_c^2 = \sum_p \frac{X_{cp}} {\sum_{c'} \frac{X_{c'p}}{X_{c'}^1}}

    Parameters
    ----------
    X : numpy array.
        Represents the export matrix. Both the sum of the rows and the sum of the columns must contain non-zero entries.
    th : scalar.
        The stopping criteria.
    
    Returns
    ----------
    ECIp : list
        Economic Complexity Index Plus for the rows of X.
    PCIp : list
        Product Compleixty Index Plus for the columns of X.
    '''
    Xc = X.sum(1)
    Xp = X.sum(0)
    Xc0 = X.sum(1)
    Xp0 = (X.transpose()/Xc0).sum(1)
    C = float(len(Xc))
    P = float(len(Xp))
    if any(Xc==0)|any(Xp==0):
        raise NameError('One dimension has a null sum')
    
    ECIp0 = log(Xc0)-log((X/Xp).sum(1))
    PCIp0 = log(Xp)-log(Xp0) 
    it_count=0
    while True:
        it_count+=1
        Den = (X.transpose()/Xc0).sum(1)
        Xc1 = (X/Den).sum(1)
        norm = exp(sum(log(Xc1)*(1./C)))
        Xc1 = Xc1/norm

        Den = (X/Xp0).sum(1)
        Xp1 = (X.transpose()/Den).sum(1)
        norm = exp(sum(log(Xp1)*(1./P)))
        Xp1 = Xp1/norm
    
        ECIp1 = log(Xc1)-log((X/Xp).sum(1))
        PCIp1 = log(Xp)-log(Xp1)
    
        #if all(abs((ECIp0-ECIp1)/ECIp0)<th)&all(abs((PCIp0-PCIp1)/PCIp0)<th):
        if (mean(abs((ECIp0-ECIp1)/ECIp0))<th)&(mean(abs((PCIp0-PCIp1)/PCIp0))<th):
            PCIp = PCIp1[:]
            ECIp = ECIp1[:]
            break
        if it_count>=max_iter:
            print('No convergence after '+str(max_iter)+' iterations')
            print mean(abs((ECIp0-ECIp1)/ECIp0)),mean(abs((PCIp0-PCIp1)/PCIp0))
            PCIp = PCIp1[:]
            ECIp = ECIp1[:]
            break
        PCIp0 = PCIp1[:]
        ECIp0 = ECIp1[:]
        Xc0 = Xc1[:]
        Xp0 = Xp1[:]
    ECIp = exp(ECIp)
    ECIp = (ECIp-mean(ECIp))/std(ECIp)
    PCIp = -exp(PCIp) #This is a modification to the original formula 
    PCIp = (PCIp-mean(PCIp))/std(PCIp)
    return ECIp.tolist(),PCIp.tolist()

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
    net_ = deepcopy(net)
    s = net_.columns.values[0] if s is None else s
    t = net_.columns.values[1] if t is None else t
    B = net_[s]>net_[t]
    if B.sum() !=0:
        S,T = zip(*net_[B][[s,t]].values)
        net_.loc[B,s]=T
        net_.loc[B,t]=S
    return net_

def unflatten(dis,s=None,t=None):
    """
    Adds repeated rows by flipping the source and the target.

    Parameters
    ----------
    dis : pandas.DataFrame
        Table with source,target plus other columns that will be copied
    s,t : str (optional)
        Labels on dis to use as source and target

    Returns
    -------
    new : pandas.DataFrame
    """
    s = dis.columns.values[0] if s is None else s
    t = dis.columns.values[1] if t is None else t
    new = concat([dis,dis.rename(columns={s:t,t:s})]).drop_duplicates()
    new = new[dis.columns.values.tolist()]
    return new

def flatten(dis,s=None,t=None,group=None,agg_method='sum'):
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
        Can be 'sum', 'max', or 'avg'
    group : list (optional)
        Columns to regard as different networks. 
        These will not be aggregated, but used as extra labels for grouping.
        For example, this can be the label of the time column.

    Returns
    -------
    new : pandas.DataFrame

    """
    s = dis.columns.values[0] if s is None else s
    t = dis.columns.values[1] if t is None else t
    group = [] if group is None else group

    new = order_columns(deepcopy(dis),s=s,t=t)
    aggcol = set(new.columns.values).difference(set(group))
    if len(aggcol) >0:
        if agg_method=='sum':
            new = new.groupby(group+[s,t]).sum().reset_index()
        elif agg_method=='max':
            new = new.groupby(group+[s,t]).max().reset_index()
        elif agg_method=='avg':
            new = new.groupby(group+[s,t]).mean().reset_index()
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
            print 'Avg deg:',2*len(T.edges())/float(len(T.nodes()))
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

def calculateRCA_by_year(data,y='',c='',p='',x='',shares=False, log_terms = False):
    '''
    This function handles input data from more than one year.
    Returns the RCA expressed in data. All RCA values belong to a country-product-year.
    Parameters
    ----------
    data : pandas.DataFrame
        Raw data. It has year,source,target,volume (trade, number of people etc.).
    y,c,p,x : str (optional)
        Labels of the columns in data used for source,target,volume
    shares : boolean (False)
        If True it will also return the shares used to calculate the RCA
    log_terms: boolean(False)
        If True it instead returns the log exports log(x), log 'baseline term' log(\sum_c x_{cpy}  \sum_p x_{cpy} / \sum_c \sum_p x_{cpy})
    and log(RCA), which is by definition the diference of these two.
    Returns
    -------
    RCA : pandas.DataFrame
        Table with the RCAs, with the columns c,p,x,RCA
        If shares is True it includes:
            s_c : Share of X_cp over X_c
            s_p : Share of X_cp over X_p
        If log_terms is True, it instead includes:
            log(x) : log of exports
            T : log of the baseline term, which is market size of product * market size of country / total world trade
            log(RCA) : log of RCA computed as log(x) - T
            
    '''
    y = data.columns.values[0] if y == '' else y
    c = data.columns.values[1] if c == '' else c
    p = data.columns.values[2] if p == '' else p
    x = data.columns.values[3] if x == '' else x
    data_ = data[[y,c,p,x]]
    
    data_ = merge(data_,data_.groupby([c,y]).sum()[[x]].rename(columns={x:x+'_'+c+'_'+y}).reset_index()
               ,how='inner',left_on=[y,c],right_on=[y,c]) #This is Tc
    data_ = merge(data_,data_.groupby([p,y]).sum()[[x]].rename(columns={x:x+'_'+p+'_'+y}).reset_index()
                  ,how='inner',left_on=[y,p],right_on=[y,p])
    data_ = merge(data_,data_.groupby(y).sum()[[x]].rename(columns={x:x+'_'+y}).reset_index()
                  ,how='inner',left_on=y,right_on=y)

    data_['RCA'] = (data_[x].astype(float)/data_[x+'_'+p+'_'+y].astype(float))/(data_[x+'_'+c+'_'+y].astype(float)/data_[x+'_'+y].astype(float))

    if shares:
        data_['s_'+c] = (data_[x].astype(float)/data_[x+'_'+c+'_'+y].astype(float)) 
        data_['s_'+p] = (data_[x].astype(float)/data_[x+'_'+p+'_'+y].astype(float))
        return data_[[y,c,p,x,'RCA','s_'+c,'s_'+p]]
    if log_terms:
        data_['log(x)'] = log10(data_[x].astype(float))
        data_['T'] = -log10((1/data_[x+'_'+p+'_'+y].astype(float))/(data_[x+'_'+c+'_'+y].astype(float)/data_[x+'_'+y].astype(float)))
        data_['log(RCA)'] = data_['log(x)'] - data_['T']
        return data_[[y,c,p,x,'RCA','log(x)','T','log(RCA)']]
    return data_[[y,c,p,x,'RCA']]


def calculatepRCA(data, y ='',c='',p='',x='', return_knn = False):
    '''
    Returns the pRCA from data. pRCA is the probability that (RCA_{y+1} > 1) given the volume of exports (x_{cpy}),
    and the 'baseline term' (\sum_c x_{cpy}  \sum_p x_{cpy} / \sum_c \sum_p x_{cpy}).
    It is computed using k-nearest neighbors, in the space of log exports and log baseline term.
    Parameters
    ----------
    data : pandas.DataFrame
        Raw data. It has source,target,volume (trade, number of people etc.).
    y,c,p,x : str (optional)
        Labels of the columns in data used for source,target,volume
    return_knn: Boolean. True if you want the knn object returned as well.
    Returns
    -------
    pRCA : pandas.DataFrame
        Table with the pRCAs, with the columns c,p,y,pRCA
        If return_knn is True the function returns this dataframe and the 
    fitted knn object for making predictions. 
    '''
    df = calculateRCA_by_year(data,y ='year',c='ccode',p='pcode',x='x',log_terms = True)
        
    #Compute (RCA > 1) next year and merge it
    df_ = df.copy()
    df_['year'] = df_['year'] - 1
    df_['RCA_y+1'] = (df_['log(RCA)'] > 0).astype(int)
    df_ = df_[['year','ccode','pcode','RCA_y+1']]
    df = df.merge(df_)
    
    #Prepare dataset for knn and fit
    M = df[['log(x)','T','RCA_y+1']].as_matrix()
    X, y = M[:,:2], M[:, 2] 
    knn = neighbors.KNeighborsRegressor(n_neighbors = 200, weights = 'uniform').fit(X, y)

    #To avoid memory error, compute predictions in split X. Predictions are output pRCA
    pRCA = array([])
    for x in array_split(X, 10):
        pRCA = append(pRCA, knn.predict(x))
    df['pRCA'] = pRCA

    if return_knn:
        return df, knn
    else:
        return df


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


def r_test(Nhops,th=0,p_val=False,directed=True,s=None,t=None,n=None):    
    '''
    Calculates the correlation and tests its significance.

    Parameters
    ----------
    Nhops : pandas.DataFrame
        Table with source,target,N as columns.
    th : float (default=0)
        Threshold to test.
        It tests whether r>th.
    p_val : boolean (False)
        If True it will return the p_value for each potential link.
    directed : boolean (True)
        If False it will treat each link as an undirected link.
    s,t,n : str,str,str (optional)
        Labels for the columns of Nhops that contain the source node (s), the target node (t), and the size of the flow (n).

    Returns
    -------
    Nt : pandas.DataFrame
        Table with s,t,n,r,t,95,99,p-value where:
        r  : correlation coefficient
        t  : t-statistic
        95 : boolean whether to reject at one-sided 95% confidence (if True the link is significantly larger than th)
        99 : boolean whether to reject at one-sided 99% confidence (if True the link is significantly larger than th)
        p-value : (optional) p-value for the t-test
    '''
    s = Nhops.columns.values[0] if s is None else s
    t = Nhops.columns.values[1] if t is None else t
    n = Nhops.columns.values[2] if n is None else n
    Nt = Nhops[Nhops.columns.values]
    N = float(Nt[n].sum())
    if not directed:
        Nt = order_columns(Nt,s=None,t=None)
        Nt = Nt.groupby([s,t]).sum().reset_index()
        Ntt = concat([Nt.groupby(s)[[n]].sum().reset_index().rename(columns={s:'tag'}),Nt.groupby(t)[[n]].sum().reset_index().rename(columns={t:'tag'})]).groupby('tag').sum()[[n]].reset_index()
        Nt = merge(Nt,Ntt.rename(columns={'tag':s,n:'n_s'}))
        Nt = merge(Nt,Ntt.rename(columns={'tag':t,n:'n_t'}))
        Nt['r']=(N*Nt[n]-Nt['n_s']*Nt['n_t'])/sqrt(Nt['n_s']*Nt['n_t']*(N-Nt['n_s'])*(N-Nt['n_t']))
    else:
        Nt = merge(Nt,Nhops.groupby(t).sum()[[n]].reset_index().rename(columns={n:'Np'}))
        Nt = merge(Nt,Nhops.groupby(s).sum()[[n]].reset_index().rename(columns={n:'Nm'}))
        Nt['r']=(N*Nt[n]-Nt['Np']*Nt['Nm'])/sqrt(Nt['Np']*Nt['Nm']*(N-Nt['Np'])*(N-Nt['Nm']))
    Nt['t']=sqrt(N-2.)*(Nt['r']/sqrt(1.-Nt['r']**2)-th/sqrt(1.-th**2))
    Nt['95'] = False
    Nt['99'] = False
    Nt.loc[Nt['t']>=1.645,'95'] = True
    Nt.loc[Nt['t']>=2.326,'99'] = True
    if p_val:
        Nt['p-value']=[1.-norm.cdf(tval) for tval in Nt['t']]
        return Nt[[s,t,n,'r','t','95','99','p-value']]
    return Nt[[s,t,n,'r','t','95','99']]



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


