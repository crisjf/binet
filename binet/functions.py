from networkx import Graph,minimum_spanning_tree,number_connected_components,is_connected,connected_component_subgraphs
import json
from pandas import DataFrame,merge,concat,read_csv
from numpy import array,matrix,mean,std,log
from scipy.interpolate import interp1d
from requests import get
import io,urllib2,bz2


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


def build_connected(dis_,th,s=None,t=None,w=None,directed=False,progress=True):
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
    s = dis_.columns.values[0] if s is None else s
    t = dis_.columns.values[1] if t is None else t
    w = dis_.columns.values[2] if w is None else w

    dis = dis_[[s,t,w]]
    dis[s] = dis[s].astype(int).astype(str)
    dis[t] = dis[t].astype(int).astype(str)

    G = Graph()
    G.add_edges_from(list(set([tuple(set(edge)) for edge in zip(dis[s],dis[t])])))
    if not is_connected(G):
        raise NameError('The provided network is not connected.')

    if not directed:
        G = Graph()    
        G.add_edges_from(zip(dis[s].values,dis[t].values,[{'weight':f} for f in dis[w]]))
        T = minimum_spanning_tree(G)
        T.add_edges_from([(u,v,{'weight':we}) for u,v,we in dis[dis[w]>=th].values.tolist()])
        if progress:
            print 'N edges:',len(T.edges())
            print 'N nodes:',len(T.nodes())
        out = []
        for u,v in T.edges():
            out.append((u,v,T.get_edge_data(u, v)['weight']))
        edges = DataFrame(out,columns=[s,t,w])
        edges[s] = edges[s].astype(dis_.dtypes[s])
        edges[t] = edges[t].astype(dis_.dtypes[t])
        return edges
    else:
        net = dis[dis[w]>=th]
        G = Graph()
        G.add_edges_from(list(set([tuple(set(edge)) for edge in zip(net[s],net[t])])))
        N_con = number_connected_components(G)
        while N_con>1:
            Gc = max(connected_component_subgraphs(G), key=len)
            data_g = [merge(DataFrame(Gc.nodes(),columns=['node_id']),dis,how='inner',left_on='node_id',right_on=s),
                  merge(DataFrame(Gc.nodes(),columns=['node_id']),dis,how='inner',left_on='node_id',right_on=t)]
            data_g = concat(data_g).drop('node_id',1).drop_duplicates()
            graphs = connected_component_subgraphs(G)
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
            G = Graph()
            G.add_edges_from(list(set([tuple(set(edge)) for edge in zip(net[s],net[t])])))
            N_con = number_connected_components(G)
        if progress:
            print 'N edges:',len(net)
            print 'N nodes:',len(set(net[s].values)|set(net[t].values))
        net[s] = net[s].astype(dis_.dtypes[s])
        net[t] = net[t].astype(dis_.dtypes[t])
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


def WDI_get(var_name,start_year=None,end_year=None):
    """Retrieves the given development indicator from the World Bank website.
    http://data.worldbank.org/data-catalog/world-development-indicators
    """
    start_year = 1995 if start_year is None else start_year
    end_year   = 2008 if end_year is None else end_year
    if end_year<start_year:
        raise NameError('start_year must come before end_year')
    WDI_base = 'http://api.worldbank.org/countries/all/indicators/'
    codes_url = 'https://commondatastorage.googleapis.com/ckannet-storage/2011-11-25T132653/iso_3166_2_countries.csv'
    
    WDI_api  = WDI_base+var_name+'?date='+str(start_year)+':'+str(end_year)+'&format=json'
    ccodes = read_csv(codes_url)[['ISO 3166-1 2 Letter Code','ISO 3166-1 3 Letter Code']].rename(columns={'ISO 3166-1 2 Letter Code':'ccode2','ISO 3166-1 3 Letter Code':'ccode'})
    r = get(WDI_api).json()
    pages = r[0]['pages']
    ind_name = r[1][0]['indicator']['value']
    print 'Getting "'+ind_name+'" between '+str(start_year)+' and '+str(end_year)
    data = [(entry['date'], entry['country']['id'],entry['value']) for entry in r[1]]
    for page in range(2,pages+1):
        r = get(WDI_api+'&page='+str(page)).json()
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

    Example
    ----------
    >>> world_trade,pnames,cnames = bnt.trade_data('hs96')
    '''
    if classification not in ['sitc','hs92','hs96','hs02','hs07']:
        raise NameError('Invalid classification')
    print 'Retrieving trade data for '+classification
    atlas_url = 'http://atlas.media.mit.edu/static/db/raw/'
    trade_file = {'sitc':'year_origin_sitc_rev2.tsv.bz2',
                  'hs92':'year_origin_hs92_4.tsv.bz2',
                  'hs96':'year_origin_hs96_4.tsv.bz2',
                  'hs02':'year_origin_hs02_4.tsv.bz2',
                  'hs07':'year_origin_hs07_4.tsv.bz2'}
    pname_file = {'sitc':'products_sitc_rev2.tsv.bz2',
                  'hs92':'products_hs_92.tsv.bz2',
                  'hs96':'products_hs_96.tsv.bz2',
                  'hs02':'products_hs_02.tsv.bz2',
                  'hs07':'products_hs_07.tsv.bz2'}

    print 'Downloading country names from '+atlas_url+'country_names.tsv.bz2'
    data = bz2.decompress(urllib2.urlopen(atlas_url+'country_names.tsv.bz2').read())
    cnames = read_csv(io.BytesIO(data),delimiter='\t')[['id_3char','name']].rename(columns={'id_3char':'ccode'}).dropna()
    
    print 'Downloading product names from '+atlas_url + pname_file[classification]
    data = bz2.decompress(urllib2.urlopen(atlas_url + pname_file[classification]).read())
    pnames = read_csv(io.BytesIO(data),delimiter='\t')
    pnames[classification] = pnames[classification].astype(str)
    if classification[:2] == 'hs':
        pnames['id_len'] = pnames[classification].str.len()
        pnames = pnames[pnames['id_len']<=4]
    pnames = pnames[[classification,'name']].rename(columns={classification:'pcode'}).dropna()
    pnames['pcode'] = pnames['pcode'].astype(int)
    pnames = pnames.sort_values(by='pcode')
    
    print 'Downloading trade   data  from '+atlas_url +trade_file[classification]
    data = bz2.decompress(urllib2.urlopen(atlas_url +trade_file[classification]).read())
    world_trade = read_csv(io.BytesIO(data),delimiter='\t')[['year','origin',classification,'export_val']].rename(columns={'origin':'ccode',classification:'pcode','export_val':'x'}).dropna()
    world_trade['year'] = world_trade['year'].astype(int)
    world_trade['pcode'] = world_trade['pcode'].astype(int)
    return world_trade,pnames,cnames



def build_html(nodes,edges,node_id =None,source_id =None,target_id=None,size_id=None,weight_id = None,x=None,y=None,color=None,props = None,progress=True):
    """Creates an html file with a d3plus visualization of the network from the dataframes nodes and edges.
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
        if len(set(nodes[color].values))>=15:
            raise NameError('Too many different colors, try using Jenks Natural Breaks')

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
            sd.append('"'+str(prop)+'":'+'"'+str(row[prop])+'"')
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




