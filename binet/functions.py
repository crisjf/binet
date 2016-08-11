from networkx import Graph,minimum_spanning_tree,number_connected_components
import json
from pandas import DataFrame,merge,concat,read_csv
from numpy import array,matrix,mean,std
import numpy as np
from scipy.interpolate import interp1d
import requests
import io,urllib2,bz2

def CalculateComplexity(M,th=0.0001):
    '''Calculates the Economic Complexity Index following the method of reflections presented in
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

def build_connected(dis,th,s='',t='',w='',directed=False):
    """Builds a connected network out of a set of weighted edges and a threshold.

    Parameters
    ----------
    dis : pandas DataFrame
            Contains at least three columns: source, target, and weight
    th : float
            Threshold to use for the network.
    s,t,w : (optional) strings
            Names of the columns in dis to use as source, target, and weight, respectively.
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

    """
    s = dis.columns.values[0] if s=='' else s
    t = dis.columns.values[1] if t=='' else t
    w = dis.columns.values[2] if w=='' else w

    G = nx.Graph()
    G.add_edges_from(list(set([tuple(set(edge)) for edge in zip(dis[s],dis[t])])))
    if not nx.is_connected(G):
        raise NameError('The provided network is not connected.')

    if not directed:
        G = Graph()    
        G.add_edges_from(zip(dis[s].values,dis[t].values,[{'weight':f} for f in dis[w]]))
        T = minimum_spanning_tree(G)
        T.add_edges_from([(u,v,{'weight':we}) for u,v,we in dis[dis[w]>=th].values.tolist()])
        print 'N edges:',len(T.edges())
        print 'N nodes:',len(T.nodes())
        out = []
        for u,v in T.edges():
            out.append((u,v,T.get_edge_data(u, v)['weight']))
        edges = DataFrame(out,columns=[s,t,w])
        return edges
    else:
        net = dis[dis[w]>=th]
        G = nx.Graph()
        G.add_edges_from(list(set([tuple(set(edge)) for edge in zip(net[s],net[t])])))
        N_con = nx.number_connected_components(G)
        while N_con>1:
            Gc = max(nx.connected_component_subgraphs(G), key=len)
            data_g = [merge(DataFrame(Gc.nodes(),columns=['node_id']),dis,how='inner',left_on='node_id',right_on=s),
                  merge(DataFrame(Gc.nodes(),columns=['node_id']),dis,how='inner',left_on='node_id',right_on=t)]
            data_g = concat(data_g).drop('node_id',1).drop_duplicates()
            graphs = nx.connected_component_subgraphs(G)
            for g in graphs:
                if len(g) != len(Gc):
                    d_temp = []
                    for v in g.nodes():
                        d_temp.append(data_g[(data_g[s]==v)|(data_g[t]==v)])
                    d_temp = concat(d_temp).drop_duplicates()
                    if len(d_temp) == 0:
                        print 'Not possible to connect nodes: ',g.nodes()
                    net.loc[len(net)] = d_temp.sort('t',ascending=False).iloc[0].values
                    break
            G = nx.Graph()
            G.add_edges_from(list(set([tuple(set(edge)) for edge in zip(net[s],net[t])])))
            N_con = nx.number_connected_components(G)
        print 'N edges:',len(net)
        print 'N nodes:',len(set(net[s].values)|set(net[t].values))
        return net
    

def calculateRCA(data_,c='',p='',x='',shares=False):
    '''Returns the RCA'''
    c = data_.columns.values[0] if c == '' else c
    p = data_.columns.values[1] if p == '' else p
    x = data_.columns.values[2] if x == '' else x
    data = data_[[c,p,x]]
    data = merge(data,data.groupby(c).sum()[[x]].rename(columns={x:x+'_'+c}).reset_index()
                         ,how='inner',left_on=c,right_on=c)
    data = merge(data,data.groupby(p).sum()[[x]].rename(columns={x:x+'_'+p}).reset_index()
                         ,how='inner',left_on=p,right_on=p)
    X = float(data.sum()[x])
    data['RCA'] = (data[x].astype(float)/data[x+'_'+p].astype(float))/(data[x+'_'+c].astype(float)/X)
    if shares:
        data['s_'+c] = (data[x].astype(float)/data[x+'_'+c].astype(float)) #Share of X_cp over X_c
        data['s_'+p] = (data[x].astype(float)/data[x+'_'+p].astype(float)) #Share of X_cp over X_p
        return data[[c,p,x,'RCA','s_'+c,'s_'+p]]   
    return data[[c,p,x,'RCA']]


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
    for c in set(df[by].values):
        gc = df[df[by]==c].sort_values(by=x)
        X = gc[x]
        Y = gc[y]
        f = interp1d(X, Y, kind=kind)
        xx = np.array(range(min(X),max(X)+1))
        yy = f(xx)
        interp+=zip(len(xx)*[c],xx,yy)
    return DataFrame(interp,columns=[by,x,y])


def WDI_get(var_name,start_year=None,end_year=None):
    """Retrieves the given worl indicator from the WDI website."""
    start_year = 1995 if start_year is None else start_year
    end_year   = 2008 if end_year is None else end_year
    if end_year<start_year:
        raise NameError('start_year must come before end_year')
    WDI_base = 'http://api.worldbank.org/countries/all/indicators/'
    codes_url = 'https://commondatastorage.googleapis.com/ckannet-storage/2011-11-25T132653/iso_3166_2_countries.csv'
    
    WDI_api  = WDI_base+var_name+'?date='+str(start_year)+':'+str(end_year)+'&format=json'
    ccodes = read_csv(codes_url)[['ISO 3166-1 2 Letter Code','ISO 3166-1 3 Letter Code']].rename(columns={'ISO 3166-1 2 Letter Code':'ccode2','ISO 3166-1 3 Letter Code':'ccode'})
    r = requests.get(WDI_api).json()
    pages = r[0]['pages']
    ind_name = r[1][0]['indicator']['value']
    print 'Getting "'+ind_name+'" between '+str(start_year)+' and '+str(end_year)
    data = [(entry['date'], entry['country']['id'],entry['value']) for entry in r[1]]
    for page in range(2,pages+1):
        r = requests.get(WDI_api+'&page='+str(page)).json()
        data += [(entry['date'], entry['country']['id'],entry['value']) for entry in r[1]]
    data = merge(ccodes,DataFrame(data,columns=['year','ccode2',ind_name]),how='left')[['ccode','year',ind_name]].dropna()
    data['ccode'] = data['ccode'].str.lower()
    data['year']  = data['year'].astype(int)
    try:
        data[ind_name] = data[ind_name].astype(float)
    except:
        pass
    return data

def trade_data(classification='sitc'):
    '''Downloads the world trade data from atlas.media.mit.edu
    '''
    if classification not in ['sitc','hs92']:
        raise NameError('Invalid classification')
    print 'Retrieving trade data for '+classification
    atlas_url = 'http://atlas.media.mit.edu/static/db/raw/'
    trade_file = {'sitc':'year_origin_sitc_rev2.tsv.bz2','hs92':'year_origin_hs92_4.tsv.bz2'}
    pname_file = {'sitc':'products_sitc_rev2.tsv.bz2','hs92':'products_hs_92.tsv.bz2'}

    print 'Downloading country names from '+atlas_url+'country_names.tsv.bz2'
    data = bz2.decompress(urllib2.urlopen(atlas_url+'country_names.tsv.bz2').read())
    cnames = read_csv(io.BytesIO(data),delimiter='\t')[['id_3char','name']].rename(columns={'id_3char':'ccode'}).dropna()
    
    print 'Downloading product names from '+atlas_url + pname_file[classification]
    data = bz2.decompress(urllib2.urlopen(atlas_url + pname_file[classification]).read())
    pnames = read_csv(io.BytesIO(data),delimiter='\t')
    if classification == 'hs92':
        pnames['id_len'] = pnames['hs92'].str.len()
        pnames = pnames[pnames['id_len']==4]
    pnames = pnames[[classification,'name']].rename(columns={classification:'pcode'}).dropna()
    pnames['pcode'] = pnames['pcode'].astype(int)
    
    print 'Downloading trade data from    '+atlas_url +trade_file[classification]
    data = bz2.decompress(urllib2.urlopen(atlas_url +trade_file[classification]).read())
    world_trade = read_csv(io.BytesIO(data),delimiter='\t')[['year','origin',classification,'export_val']].rename(columns={'origin':'ccode',classification:'pcode','export_val':'x'}).dropna()
    world_trade['year'] = world_trade['year'].astype(int)
    world_trade['pcode'] = world_trade['pcode'].astype(int)
    return world_trade,pnames,cnames


def build_html(nodes,edges,node_id = '',source_id = '',target_id='',**kwargs):
    """Creates an html file with a d3plus visualization of the network from the dataframes nodes and edges."""

    node_id = nodes.columns.values[0] if node_id == '' else node_id
    source_id = edges.columns.values[0] if source_id == '' else source_id
    target_id = edges.columns.values[0] if target_id == '' else target_id

    sample_data = '['
    positions = '['
    connections = '['

    for index,row in nodes.iterrows():
        sample_data+='{"name":"'+str(row[node_id]) +'", "size":'+str(np.log(row['n_entries']))+',"desc": "'+row[node_id]+'","n_entries":'+str(row['n_entries'])+'},\n' #industries
        positions += '{"name":"'+str(row[node_id]) +'", "x":'+str(row['x'])+',"y": '+str(row['y'])+'},\n' #industries
    for index,row in edges.iterrows():
        connections += '{"source": "'+str(int(row[source_id]))+'", "target": "'+str(int(row[target_id]))+'","strength":'+str(row['f'])+'},\n' #industries    

    sample_data = sample_data[:-2]+']\n'
    positions = positions[:-2]+']\n'
    connections = connections[:-2]+']\n'

    html = """<!doctype html>
    <meta charset="utf-8">
    <script src="http://www.d3plus.org/js/d3.js"></script>
    <script src="http://www.d3plus.org/js/d3plus.js"></script>
    <div id="viz"></div>
    <script>
    // create sample dataset
    var sample_data = """ 
    html += sample_data + 'var positions =' + positions + 'var connections ='+connections
    html += """var visualization = d3plus.viz()
        .container("#viz")
        .type("network")
        .data(sample_data)
        .nodes(positions)
        .edges({"value": connections,"size": "strength"})
        .size("size")
        .id("name")
        .tooltip(["desc","name", "size","n_entries"])
        .draw()
    </script>"""
    return html