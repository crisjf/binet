from networkx import Graph,minimum_spanning_tree,number_connected_components,is_connected,connected_component_subgraphs
from pandas import DataFrame,merge,concat,read_csv
from numpy import array,matrix,mean,std,log,sqrt,exp,log10,array_split,random
from scipy.interpolate import interp1d
from sklearn import neighbors
from copy import deepcopy
from itertools import chain
try:
    from community import best_partition
except:
    print 'Warning: No module named community found.'
import json


def df_shuffle(df):
    return df.reindex(random.permutation(df.index))

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

def list_join(a):
    return list(chain.from_iterable(a))

def growth(L_s,t=None,x=None,group=None,useLast=False):
    '''
    Calculates the growth of a variable in the given table.

    Parameters
    ----------
    L_s : pandas.DataFrame
        Table with columns:
            t     : discrete time column
            group : columns that index the group
            x     : variable to calculate growth of
    t : str (optional)
        Column name of discrete time variable
    group : list (optional)
        Columns to use as groups (ex: country codes)
    x : str (optional)
        Column of the variable to calcualte growth of
    useLast : boolean (False)
        If True it will use the last timestamp in the time window as the reference.
        For example, the growth between 2006 and 2007 will be counted as growth for 2007 if this variable is True

    Returns
    -------
    g : pandas.DataFrame
        Table with columns:
            t     : time column
            group : group columns
            g     : growth column
    '''
    t = L_s.columns.values[0] if t is None else t
    group = [L_s.columns.values[1]] if group is None else group
    x = L_s.columns.values[2] if x is None else x
    g = L_s[[t]+group+[x]]
    time = ['NULL']+sorted(list(set(g[t])))+['NULL']
    g[t+'m1'] = [time[time.index(tt)-1] for tt in g[t]]
    g = merge(g.drop(t+'m1',1),g.drop(t,1).rename(columns={t+'m1':t,x:x+'p1'}))
    if useLast:
        g['year'] = g['year']+1
        g = g.rename(columns={x:x+'m1',x+'p1':x})
        g['g'] = log(g[x]/g[x+'m1'])
    else:
        g['g'] = log(g[x+'p1']/g[x])
    g = g[[t]+group+['g']]
    return g


def communities(net,s=None,t=None,w=None,node_id=None,progress=True):
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
    w : str (optional)
        Column of the weight.
    progress : boolean (True)
    	If True it will print the number of communities

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
    if w is None:
        part = best_partition(G)
    else:
        part = best_partition(G,weight=w)
    part = DataFrame(part.items(),columns=[node_id,'community_id'])
    if progress:
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
        ps = list(set(net[p])|set(net[p+'p']))
        if t is not None:
            ts = list(set(net[t]))
            combs = DataFrame([(t1,p1,p2) for p1 in ps for p2 in ps for t1 in ts],columns=[t,p,p+'p'])
        else:
            combs = DataFrame([(p1,p2) for p1 in ps for p2 in ps],columns=[p,p+'p'])
        combs = combs[combs[p]!=combs[p+'p']]
        net = merge(combs,net,how='left').fillna(0)
        net = merge(net,data_)
        net = merge(net,data_.rename(columns={p:p+'p','N_'+p:'N_'+p+'p'}))
    return net



import statsmodels.api as sm
def residualNet(data,s=None,t=None,x=None,g=None,numericalControls=[],categoricalControls=[],addDummies=False,show_results=False):
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
    addDummies : boolean (False)
        If True it will add controls for each node.
    show_results : boolean (False)
        If True it will print the R2 and the values of the coefficients
        
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
            if show_results:
                print 'R2 for',x,'with dummies' if addDummies else 'without dummies','for '+str(gg),model.rsquared
                for columns_name,parameter_value in zip((['c']+list(set(numericalControls))+list(set(_categoricalControls))),model.params):
                    print '\t',columns_name,parameter_value
            out.append(data_g[[g,s,t,x+'_res',x]+list(set(numericalControls))+list(set(categoricalControls))])
        data_ = concat(out)[[g,s,t,x+'_res',x]+list(set(numericalControls))+list(set(categoricalControls))]
    else:
        Y = data_[x].values
        X = data_[list(set(numericalControls))+list(set(_categoricalControls))].values
        X = sm.add_constant(X)
        model = sm.OLS(Y,X).fit()
        data_[x+'_res'] = Y-model.predict(X)
        data_ = data_[[s,t,x+'_res',x]+list(set(numericalControls))+list(set(categoricalControls))]
        if show_results:
            print 'R2 for',x,'with dummies' if addDummies else 'without dummies',model.rsquared
            for columns_name,parameter_value in zip((['c']+list(set(numericalControls))+list(set(_categoricalControls))),model.params):
                print '\t',columns_name,parameter_value
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

def flatten(dis,s=None,t=None,group=None,agg_method='avg'):
    """
    Orders the nodes such that the source codes are always smaller than the target codes, and aggregates.
    It is used to flatten a directed network in a table.

    Parameters
    ----------
    dis : pandas.DataFrame
        Table with source,target plus other columns that will be aggregated
    s,t : str (optional)
        Labels on dis to use as source and target
    agg_method : str (default='avg')
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


def build_connected(net,th,s=None,t=None,w=None,directed=False,mst2max=False):
    """Builds a connected network out of a set of weighted edges and a threshold.

    Parameters
    ----------
    net : pandas.DataFrame
        Contains at least three columns: source, target, and weight
    th : float
        Threshold to use for the network.
    s,t,w : str (optional)
        Names of the columns in net to use as source, target, and weight, respectively.
        If it is not provided, the first three columns as assumed to be s,t,and w, respectively.
    directed : boolean (False)
        Wether to consider the network as directed or as undirected.
    mst2max : boolean (False)
        If True it will set all the links from the MST to the maximum possible value.

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
        mst_edges = T.edges()
        T.add_edges_from([(u,v,{'weight':-we}) for u,v,we in dis[dis[w]>=th].values.tolist()])
        if mst2max:
            w_max = dis[w].max() 
            for u,v in mst_edges:
                T[u][v]['weight']=-w_max
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
        edges_mst = Tedges[idx][[s,t]]
        edges_out = merge(concat([dis[dis[w]>=th][[s,t]],edges_mst]).drop_duplicates(),dis)
        if mst2max:
            w_max = dis[w].max()
            edges_out.loc[(edges_out[s].isin(set(edges_mst[s])))&(edges_out[t].isin(set(edges_mst[t]))),w] = w_max
        edges_out[s] = edges_out[s].astype(net.dtypes[s])
        edges_out[t] = edges_out[t].astype(net.dtypes[t])
        return edges_out


def calculateRCA(data,y=None,c=None,p=None,x=None,shares=False,log_terms=False):
    '''
    Returns the RCA expressed in data

    Parameters
    ----------
    data : pandas.DataFrame
        Raw data:
            y : time (year for example, only if parameter y is provided)
            c : source (country for example)
            p : target (product for example)
            x : volume (trade volume for example)
    y : str (optional)
        Label of the year column. 
        If not provided it will pool everything together.
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
    if (y is None):
        c = data.columns.values[0] if c is None else c
        p = data.columns.values[1] if p is None else p
        x = data.columns.values[2] if x is None else x
    elif (data.columns.values[0]!=y):
        c = data.columns.values[0] if c is None else c
        p = data.columns.values[1] if p is None else p
        x = data.columns.values[2] if x is None else x
    else:
        c = data.columns.values[1] if c is None else c
        p = data.columns.values[2] if p is None else p
        x = data.columns.values[3] if x is None else x
    if y is not None:
        if c==y:
            raise NameError('Please provide parameter c')
        elif len(set([c,p,x,y])) < 4:
            raise NameError("Column labels were confused")

    if y is None:
        data_ = data[[c,p,x]].groupby([c,p]).sum().reset_index()
        data_ = merge(data_,data_.groupby(c).sum()[[x]].rename(columns={x:x+'_'+c}).reset_index(),how='inner',left_on=c,right_on=c)
        data_ = merge(data_,data_.groupby(p).sum()[[x]].rename(columns={x:x+'_'+p}).reset_index(),how='inner',left_on=p,right_on=p)
        data_[x+'_y'] = float(data_.sum()[x])
        out_cols = [c,p,x,'RCA']    
    else:
        data_ = data[[y,c,p,x]].groupby([y,c,p]).sum().reset_index()
        data_ = merge(data_,data_.groupby([y,c]).sum()[[x]].rename(columns={x:x+'_'+c}).reset_index(),how='inner',left_on=[y,c],right_on=[y,c])
        data_ = merge(data_,data_.groupby([y,p]).sum()[[x]].rename(columns={x:x+'_'+p}).reset_index(),how='inner',left_on=[y,p],right_on=[y,p])
        data_ = merge(data_,data_.groupby([y]).sum()[[x]].rename(columns={x:x+'_y'}).reset_index(),how='inner',left_on=y,right_on=y)        
        out_cols = [y,c,p,x,'RCA']
    data_['RCA'] = (data_[x].astype(float)/data_[x+'_'+p].astype(float))/(data_[x+'_'+c].astype(float)/data_[x+'_y'].astype(float))        
    if shares:
        data_['s_'+c] = (data_[x].astype(float)/data_[x+'_'+c].astype(float)) 
        data_['s_'+p] = (data_[x].astype(float)/data_[x+'_'+p].astype(float))
        out_cols += ['s_'+c,'s_'+p]
    if log_terms:
        data_['log(x)'] = log10(data_[x].astype(float))
        data_['T'] = -log10((1/data_[x+'_'+p].astype(float))/(data_[x+'_'+c].astype(float)/data_[x+'_y'].astype(float)))
        data_['log(RCA)'] = data_['log(x)'] - data_['T']
        out_cols += ['log(x)','T','log(RCA)']
    return data_[out_cols]


def calculateRCApop(data,pop,y=None,c=None,p=None,x=None,P=None,shares=False):
    '''
    Returns the RCA adjusted by population

    Parameters
    ----------
    data : pandas.DataFrame
        Raw data:
            y : time (year for example, only if parameter y is provided)
            c : source (country for example)
            p : target (product for example)
            x : volume (trade volume for example)
    pop : pandas.DataFrame
        Raw population data. The column label for the region must coincide with the label in data.
            y : time (year for example, only if paramter y is provided)
            c : source (country for example)
            P : population data
    y : str (optional)
        Label of the year column. 
        If not provided it will pool everything together.
    c,p,x,P : str (optional)
        Labels of the columns in data used for source,target,volume, and population
    shares : boolean (False)
        If True it will also return the shares used to calculate the RCA

    Returns
    -------
    RCApop : pandas.DataFrame
        Table with the RCApops, with the columns c,p,x,RCApop
        If shares is True it also includes:
            s_c : Share of X_cp over Pop_c
            s_p : Share of X_p over Pop
    '''

    if (y is None):
        c = data.columns.values[0] if c is None else c
        p = data.columns.values[1] if p is None else p
        x = data.columns.values[2] if x is None else x
    elif (data.columns.values[0]!=y):
        c = data.columns.values[0] if c is None else c
        p = data.columns.values[1] if p is None else p
        x = data.columns.values[2] if x is None else x
    else:
        c = data.columns.values[1] if c is None else c
        p = data.columns.values[2] if p is None else p
        x = data.columns.values[3] if x is None else x
    if c not in pop.columns.values:
        raise NameError("Column "+c+" not found in population table")
        
    if (y is None):
        P = pop.columns.values[1] if P is None else P
    elif pop.columns.values[0]!=y:
        P = pop.columns.values[1] if P is None else P
    else:
        P = pop.columns.values[2] if P is None else P    
    if P==c:
        raise NameError('Please provide parameter P')

    if y is not None:
        if c==y:
            raise NameError('Please provide parameter c')

    if len(set(data[c]).difference(set(pop[c])))!=0:
        print 'Warning: missing population for '+str(len(set(data[c]).difference(set(pop[c]))))+' regions'
    if len(set([c,p,x,P,y])) < 5:
        raise NameError("Column labels were confused")

    if y is None:
        data_ = data[[c,p,x]].groupby([c,p]).sum().reset_index()
        pop_ = pop[[c,P]].groupby(c).mean().reset_index()
    else:
        data_ = data[[y,c,p,x]].groupby([y,c,p]).sum().reset_index()
        pop_ = pop[[y,c,P]].groupby([y,c]).mean().reset_index()
    
    data_ = merge(data_,pop_,how='inner')
    if y is None:
        data_ = merge(data_,data_.groupby(p).sum()[[x]].rename(columns={x:x+'_'+p}).reset_index(),how='inner',left_on=p,right_on=p)
        data_[P+'_y'] = data_[[c,P]].drop_duplicates()['pop'].sum()
        out_cols = [c,p,x,'RCApop']
    else:
        data_ = merge(data_,data_.groupby([y,p]).sum()[[x]].rename(columns={x:x+'_'+p}).reset_index(),how='inner',left_on=[y,p],right_on=[y,p])
        data_ = merge(data_,data_[[y,c,P]].drop_duplicates().groupby([y]).sum()[[P]].rename(columns={P:P+'_y'}).reset_index(),how='inner',left_on=[y],right_on=[y])
        out_cols = [y,c,p,x,'RCApop']
    data_['RCApop'] = (data_[x].astype(float)/data_[P].astype(float))*(data_[P+'_y'].astype(float)/data_[x+'_'+p].astype(float))
    if shares:
        data_['s_'+c] = data_[x].astype(float)/data_[P].astype(float)
        data_['s_'+p] = data_[x+'_'+p].astype(float)/data_[P+'_y'].astype(float)
        out_cols += ['s_'+c,'s_'+p]
    return data_[out_cols]



def calculateRCA_by_year(data,y='',c='',p='',x='',shares=False, log_terms = False):
    '''
    This function handles input data from more than one year.
    Returns the RCA expressed in data. All RCA values belong to a country-product-year.
    
    This function will be deprecated in favor of the new functionality of calculateRCA(). 
    Please see the documentation for calculateRCA().

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
    print 'Warning: This function will be deprecated in favor of the new functionality of calculateRCA().\nPlease see the documentation for calculateRCA().'
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
    print 'Warning: using new yearly RCA funcion. Please check results before using.'
    # df = calculateRCA_by_year(data,y ='year',c='ccode',p='pcode',x='x',log_terms = True)
    df = calculateRCA(data,y ='year',c='ccode',p='pcode',x='x',log_terms = True)
        
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
    '''
    Groups df by the given column, and interpolates the missing values.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to be interpolated. It has the columns
            by : group (e.g. country id)
            x  : independent variable (e.g. year)
            y  : dependent variable (e.g. population)
    by,x,y: str (optional)
        Label of the columns for the groups, independent variable, and dependent variable.
        If not provided, it will take first, second, and third, respeectively.
    kind : str (optional)
        Default is linear.
        Type of interpolation, see documentation for scipy.interpolate.interp1d()
    
    Returns
    -------
    interp : pandas.DataFrame
        Table with the interpolated results.
    '''
    by = df.columns.values[0] if by is None else by
    x  = df.columns.values[1] if x is None else x
    y  = df.columns.values[2] if y is None else y
    interp = []
    cs = list(set(df[by].values))
    for c in cs:
        gc = df[df[by]==c].sort_values(by=x).dropna()
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

def read_cyto(cyto_file,node_id='shared_name'):
    """Reads the positions of the nodes in cytoscape"""
    with open(cyto_file) as data_file:
        cyto = json.load(data_file)
    out = []
    for node in cyto['elements']['nodes']:
        out.append( (node['data']['shared_name'],node['position']['x'],node['position']['y']))
    return DataFrame(out,columns=[node_id,'x','y'])


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

