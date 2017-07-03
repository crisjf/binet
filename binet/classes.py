from pandas import DataFrame,merge,concat
from collections import defaultdict
from numpy import sqrt,mean,corrcoef,log
from networkx import Graph,set_node_attributes,set_edge_attributes
from functions import calculateRCA,CalculateComplexity,build_connected,build_html,CalculateComplexityPlus
from functions_gt import get_pos
from os import getcwd
import webbrowser
from copy import deepcopy
from itertools import chain 

class gGraph(Graph):
    def __init__(self,node_id=None):
        '''
        Wrapper for nx.Graph() class that integrates it with pandas.
        '''
        self.node_id = 'node_id' if node_id is None else node_id
        super(gGraph,self).__init__()

    def __str__(self):
        out = ''
        out+= 'Nodes: '+self.node_id + ' (' + str(len(self.nodes())) + ')'
        return out.encode('utf-8')

    def edges(self,as_df=False,data=False):
        edges = []
        if data&(not as_df):
            return super(gGraph,self).edges(nbunch=None, data=True, default=None)
        elif as_df:
            props = set([])
            edges = super(gGraph,self).edges(nbunch=None, data=True, default=None)
            for u,v,d in edges:
                props=props|set(d.keys())
            props = list(props)
            es = []
            for u,v,d in edges:
                d = defaultdict(lambda:'NA',d)
                es.append([u,v]+[d[p] for p in props])
            return DataFrame(es,columns=[str(self.node_id)+'_x',str(self.node_id)+'_y']+props)
        else:
            return super(gGraph,self).edges(nbunch=None, data=False, default=None)            

    def nodes(self,as_df=False,data=False):
        '''
        Returns a list of nodes.
        If ad_df is set to True it will return the nodes properties as well.

        Parameters
        ----------
        as_df : boolean (False)
            If True it will return the nodes as a DataFrame.
        
        Returns
        -------
        nodes : list or pandas.DataFrame
            List of nodes or DataFrame with nodes and properties.
        '''
        if not as_df:
            return super(gGraph,self).nodes(data=data)
        else:
            nns = super(gGraph,self).nodes(data=True)
            properties = list(set(chain.from_iterable([d.keys() for u,d in nns])))
            out = []
            for u,d in nns:
                oout = [u]
                for prop in properties:
                    try:
                        oout+=[d[prop]]
                    except:
                        oout+=['NA']
                out.append(oout)

            return DataFrame(out,columns=[self.node_id]+properties)

    def degree(self,nbunch=None,as_df=False,weighted=False):
        '''
        Return the degree of a node or nodes.

        The node degree is the number of edges adjacent to that node.

        Parameters
        ----------
        nbunch : iterable container, optional (default=all nodes)
            A container of nodes.  The container will be iterated
            through once.

        as_df : boolean (False)
            If True it will return the a pandas.DataFrame

        weighted : boolean (False)
            If True, it will return a the weighted degree using 'weight' as the weight. 

        Returns
        -------
        nd : dictionary or pandas.DataFrame
            A dictionary with nodes as keys and degree as values.
            If as_df=True it returns a pandas.DataFrame with:
                node_id : Node identifier
                degree : Unweighted degree
                degree_w : Weighted degree
        '''
        if as_df:
            deg = super(gGraph,self).degree(nbunch=nbunch).items()
            deg = DataFrame(deg,columns=[self.node_id,'degree'])
            if weighted:
                deg_w = super(gGraph,self).degree(nbunch=nbunch,weight='weight').items()
                deg_w = DataFrame(deg_w,columns=[self.node_id,'degree_w'])
                return merge(deg,deg_w)
            else:
                return deg
        else:
            if weighted:
                return super(gGraph,self).degree(nbunch=nbunch,weight='weight')
            else:
                return super(gGraph,self).degree(nbunch=nbunch).items()





class BiGraph(Graph):
    def __init__(self,side=0,aside=1):
        """
        Base class for undirected bipartite graphs.
        Based on networkx Graph class, but adds a property called 'side' to each node, and modifies the methods to make the access of each side easier.

        Parameters
        ----------
        side,aside : int or str
            Tags for each side of the bipartite network.
        """
        super(BiGraph,self).__init__()
        self.side  = side
        self.aside = aside
        self.P = {side:None,aside:None}
        self._ptype = 'CP'
    
    def add_nodes_from(self,nodes,side):
        if not hasattr(nodes[0], '__iter__'):
            nns = []
            for u in nodes:
                d = {'side':side}
                nns.append((u,d))
        else:   
            if len(nodes[0]) == 2:
                nns = []
                for u,d in nodes:
                    if 'side' not in d.keys():
                        d['side'] = side
                    nns.append((u,d))
            else:
                raise NameError('Wrong input format')
        super(BiGraph,self).add_nodes_from(nns)

    def add_edges_from(self,edges):
        self.add_nodes_from([u for u,v in edges],self.side)
        self.add_nodes_from([v for u,v in edges],self.aside)
        super(BiGraph,self).add_edges_from(edges)
    
    def nodes(self,side,as_df=False):
        '''
        Returns a list of nodes.
        If ad_df is set to True it will return the nodes properties as well.

        Parameters
        ----------
        side : int or str
            Tags for each side of the bipartite network.
        as_df : boolean (False)
            If True it will return the nodes as a DataFrame.
        
        Returns
        -------
        nodes : list or pandas.DataFrame
            List of nodes or DataFrame with nodes and properties.
        '''
        self._check_side(side)
        if not as_df:
            return [u for u in self.nodes_iter() if self.node[u]['side'] == side]
        else:
            nns = super(BiGraph,self).nodes(data=True)
            properties = list(set(chain.from_iterable([d.keys() for u,d in nns if d['side']==side])))
            properties.remove('side')
            out = []
            for u,d in nns:
                if d['side'] == side:
                    oout = [u]
                    for prop in properties:
                        try:
                            oout+=[d[prop]]
                        except:
                            oout+=['NA']
                    out.append(oout)
            return DataFrame(out,columns=[side]+properties)

    def edges(self,nbunch=None, data=False, default=None,as_df=False):
        '''
        Nodes go from side to aside.
        '''
        nodes_side = set(self.nodes(self.side))
        edges = []
        if as_df|data:
            props = set([])
            for u,v,d in super(BiGraph,self).edges(nbunch=None, data=True, default=None):
                if u in nodes_side:
                    edges.append((u,v,d))
                else:
                    edges.append((v,u,d))
                props=props|set(d.keys())
        else:
            for u,v in super(BiGraph,self).edges(nbunch=None, data=False, default=None):
                if u in nodes_side:
                    edges.append((u,v))
                else:
                    edges.append((v,u))
        if not as_df:
            return edges
        else:
            props = list(props)
            es = []
            for u,v,d in edges:
                d = defaultdict(lambda:'NA',d)
                es.append([u,v]+[d[p] for p in props])
            return DataFrame(es,columns=[self.side,self.aside]+props)


    def degree(self,side,nbunch=None, weight=None,as_df=False):
        '''
        Return degree of single node or of nbunch of nodes.
        If nbunch is ommitted, then return degrees of all nodes.

        Parameters
        ----------
        side   : int or str
            Tags for each side of the bipartite network.
        nbunch : list or node id (default=all nodes)
            Nodes to get the degree of.
        weight : str
            Edge property to use as weight.
        as_df  : boolean (False)
            If True it returns the degrees in a pandas DataFrame.

        Returns
        -------
        k : dictionary or DataFrame
            Dictionary with node id as keys or DataFrame with two columns (when as_df is set to True)
        '''
        self._check_side(side)
        nbunch = self.nodes(side) if nbunch is None else list(set(self.nodes(side)).intersection(nbunch))
        k = super(BiGraph,self).degree(nbunch=nbunch,weight=weight)
        if not as_df:
            return k
        else:
            return DataFrame(k.items(),columns=[side,'degree'])

    def _check_side(self,side):
        """Returns an error when the requested side is not found in the network."""
        if side not in [self.aside,self.side]:
            raise NameError('Wrong side label, choose between '+str(self.side)+' and '+str(self.aside))

    def set_node_attributes(self,side,name,values):
        '''
        Set node of type side attributes from dictionary of nodes and values.
        Only sets one attribute at a time.

        Parameters
        ----------
        name  : string
            Attribute name
        side  : int or str
            Tags for each side of the bipartite network.
        values: dict
            Dictionary of attribute values keyed by node. 
            Nodes that do not belong to side will not be considered.
        '''
        self._check_side(side)
        ns   = set(values.keys()).intersection(set(self.nodes(side)))
        vals = {val:values[val] for val in ns}
        set_node_attributes(self,name,vals)

    def set_edge_attributes(self,name,values):
        """Set edge attributes from dictionary of edge tuples and values.

        Parameters
        ----------
        name : string
            Attribute name
        values : dict
            Dictionary of attribute values keyed by edge (tuple). 
            The keys must be tuples of the form (u, v). 
            If values is not a dictionary, then it is treated as a single attribute value that is then applied to every edge.
        """
        set_edge_attributes(self,name,values)

    def _project_CP(self,side):
        """
        Builds the projection of the bipartite network on to the chosen side.
        The projection is done using conditional probability.

        Parameters
        ----------
        side : int or str
            Tags for each side of the bipartite network.
        """
        self._check_side(side)
        aside = self.side if side == self.aside else self.aside
        net = self.edges(as_df=True)[[side,aside]]
        dis = merge(net,net,how='inner',left_on=aside,right_on=aside).groupby([side+'_x',side+'_y']).count().reset_index().rename(columns={aside:'n_both'})
        nodes = merge(self.nodes(side,as_df=True)[[side]].reset_index().rename(columns={'index':side+'_index'}),DataFrame(self.degree(side).items(),columns=[side,'n']))
        dis = merge(dis,nodes,how='left',right_on=side,left_on=side+'_x').drop(side,1)
        dis = merge(dis,nodes,how='left',right_on=side,left_on=side+'_y').drop(side,1)
        dis = dis[dis[side+'_index_x']>dis[side+'_index_y']].drop([side+'_index_x',side+'_index_y'],1)

        dis['p_x'] = dis['n_both']/dis['n_x'].astype(float)
        dis['p_y'] = dis['n_both']/dis['n_y'].astype(float)
        dis['fi'] = dis[['p_x','p_y']].min(1)
        dis = dis[[side+'_x',side+'_y','fi']]
        self.P[side] = gGraph(node_id=side)
        self.P[side].add_weighted_edges_from([val[1:] for val in dis.itertuples()])
        nodes = merge(self.P[side].nodes(as_df=True),self.nodes(side,as_df=True),how='left')
        properties = nodes.columns.values.tolist()
        properties.remove(side)
        for prop in properties:
            values = dict(zip(nodes[side].values,nodes[prop].values))
            set_node_attributes(self.P[side],prop,values)

    def _project_AA(self,side):
        """
        Builds the projection of the bipartite network on to the chosen side.
        The projection is done using the ADAMIC-ADAR index.

        Parameters
        ----------
        side : int or str
            Tags for each side of the bipartite network.
        """
        self._check_side(side)
        aside = self.side if side == self.aside else self.aside
        net = self.edges(as_df=True)[[side,aside]]
        AA = merge(net,net,how='inner',left_on=aside,right_on=aside)
        nodes = self.nodes(side,as_df=True)[[side]].reset_index().rename(columns={'index':side+'_index'})

        AA = merge(AA,nodes.rename(columns={side:side+'_x',side+'_index':side+'_index_x'}),how='left',right_on=side+'_x',left_on=side+'_x')
        AA = merge(AA,nodes.rename(columns={side:side+'_y',side+'_index':side+'_index_y'}),how='left',right_on=side+'_y',left_on=side+'_y')
        AA = AA[AA[side+'_index_x']>AA[side+'_index_y']].drop([side+'_index_x',side+'_index_y'],1)
        AA = merge(AA,self.degree(aside,as_df=True))
        AA['AA'] = 1./log(AA['degree'])
        AA = AA[[side+'_x',side+'_y','AA']].groupby([side+'_x',side+'_y']).sum().reset_index()

        self.P[side] = gGraph(node_id=side)
        self.P[side].add_weighted_edges_from([val[1:] for val in AA.itertuples()])
        nodes = merge(self.P[side].nodes(as_df=True),self.nodes(side,as_df=True),how='left')
        properties = nodes.columns.values.tolist()
        properties.remove(side)
        for prop in properties:
            values = dict(zip(nodes[side].values,nodes[prop].values))
            set_node_attributes(self.P[side],prop,values)

    def _project_NK(self,side):
        """
        Builds the projection of the bipartite network on to the chosen side.
        The projection is done using the NK conditional probability.

        Parameters
        ----------
        side : int or str
            Tags for each side of the bipartite network.
        """
        self._check_side(side)
        aside = self.side if side == self.aside else self.aside

        E = self.recombination_ease(aside)
        net = self.edges(as_df=True)[[side,aside]]
        nodes = merge(E,net).groupby(side).sum()[['E']].reset_index().reset_index().rename(columns={'index':side+'_index'})

        dis = merge(E,merge(net,net,how='inner',left_on=aside,right_on=aside))
        dis = dis.groupby([side+'_x',side+'_y']).sum()[['E']].reset_index().rename(columns={'E':'E_both'})
        dis = merge(dis,nodes,how='left',right_on=side,left_on=side+'_x').drop(side,1)
        dis = merge(dis,nodes,how='left',right_on=side,left_on=side+'_y').drop(side,1)
        dis = dis[dis[side+'_index_x']>dis[side+'_index_y']].drop([side+'_index_x',side+'_index_y'],1)
        dis['p_x'] = dis['E_both']/dis['E_x'].astype(float)
        dis['p_y'] = dis['E_both']/dis['E_y'].astype(float)
        dis['fi'] = dis[['p_x','p_y']].min(1)
        dis = dis[[side+'_x',side+'_y','fi']]
        
        self.P[side] = gGraph(node_id=side)
        self.P[side].add_weighted_edges_from([val[1:] for val in dis.itertuples()])
        nodes = merge(self.P[side].nodes(as_df=True),self.nodes(side,as_df=True),how='left')
        properties = nodes.columns.values.tolist()
        properties.remove(side)
        for prop in properties:
            values = dict(zip(nodes[side].values,nodes[prop].values))
            set_node_attributes(self.P[side],prop,values)

    def projection(self,side,as_df=False,ptype=None):
        """
        Returns the network projection (will calculate if it does not exist).

        Parameters
        ----------
        side : int or str
            Tags for each side of the bipartite network.
        as_df : boolean (False)
            If True it returns the projection as a DataFrame
        ptype : str ('CP')
            Similarity index used to calculate the projection.
            Choose between conditional probability ('CP'), adamic-adamar ('AA'), and K-plexity ('NK').

        Returns
        ----------
        P : networkx.Graph() object
            Weighted betwork containing the projected network.
        """
        ptype = self._ptype if ptype is None else ptype
        if (ptype != self._ptype)|(self.P[side] is None):
            self._ptype = ptype
            if self._ptype == 'CP':
                self._project_CP(side)
            elif self._ptype == 'AA':
                self._project_AA(side)
            elif self._ptype == 'NK':
                self._project_NK(side)
            else:
                raise NameError("Unrecognized projection type.")

        if as_df:
            P = self.P[side]
            return DataFrame([(u,v,P.get_edge_data(u,v)['weight']) for u,v in P.edges()],columns=[side+'_x',side+'_y','weight'])
        else:
            return self.P[side]

    def _best_th(self,side):
        P = self.projection(side)
        W = [P.get_edge_data(u,v)['weight'] for u,v in P.edges()]
        N_edges = int(len(P.nodes())*3.3)
        th = min(sorted(W,reverse=True)[:N_edges])
        return th

    def trim_projection(self,side,th=None):
        if th is None:
            th = self._best_th(side)
        self._check_side(side)

        P = build_connected(self.projection(side,as_df=True),th,progress=False)
        return P

    def _get_projection_pos(self,side,P,C=None):
        '''This function requires graph_tool'''
        pos = get_pos(P[[side+'_x',side+'_y']],node_id=side,comms=True,progress=False,C=C)
        X = {}
        Y = {}
        Cc = {}
        for i,x,y,c in pos.values:
            X[i]=x
            Y[i]=y
            Cc[i]=c
        self.set_node_attributes(side,'x',X)
        self.set_node_attributes(side,'y',Y)
        self.set_node_attributes(side,'c',Cc)
        return pos

    def draw_projection(self,side,th=None,C=None,path='',show=True,color=True,color_name=None,jenks=None):
        '''
        Draws the projection using d3plus.
        
        Parameters
        ----------
        side : int or str
            Tag for the projectino of the bipartite network to draw.
        th : float (optional)
            If not provided, it will use a threshold such that the average degree is between 3 and 4.
        show : boolean (True)
            If True it will open a web tab to display the network.
        color_name : str
            Name of column to use for coloring.
        jenks : int (optional)
            If provided, it will apply Jenks natural breaks on the color column.


        path must be the absolut path, including the "/" symbol at the end. 
        This function requires graph_tool
        ptype='CP'
        
        '''

        max_colors=25
        if th is None:
            th = self._best_th(side)
        color_name = 'c' if color_name is None else color_name

        P = self.trim_projection(side,th=th)
        cs = P[[side+'_x',side+'_y']]
        self._get_projection_pos(side,P,C=C)

        path = 'file://'+getcwd()+'/' if path == '' else path+'/'

        props = [val for val in self.nodes(side,as_df=True).columns.values.tolist() if val not in set([side,'x','y'])]

        nodes = self.nodes(side,as_df=True)
        nodes = merge(nodes,DataFrame(self.projection(side).nodes(),columns=[side]),how='right')
        if color:
            html = build_html(nodes,cs,node_id=side,source_id=side+'_x',target_id=side+'_y',color=color_name,props=props,progress=False,max_colors=max_colors)
        else:
            html = build_html(nodes,cs,node_id=side,source_id=side+'_x',target_id=side+'_y',props=props,progress=False)
        out = side+'_'+self.name+'_th'+str(th)+'_ptype'+self._ptype+'.html' 
        open(out,mode='w').write(html.encode('utf-8'))
        if show:
            webbrowser.open(path+out)
            print 'OUT: ',path+out

class mcp(BiGraph):
    def __init__(self,c=None,p=None,name='',data=None,use=None,nodes_c=None,nodes_p=None):
        """
        Base class for undirected bipartite graphs.

        Parameters
        ----------
        data : list or pandas DataFrame
            Data to initialize graph. If data=None (default) an empty BiGraph is created.  
            The data can be a list of edges in the form [(u,v,x),(u,v,x),...], or a pandas DataFrame.
        use  : list ([c,p,x])
            If data is a pandas DataFrame, use indicates what columns to use. 
            If use=None (default) the first three columns are used.
        c,p  : string
            Name to refer to both sides of the network.
        nodes_c,nodes_p : pandas DataFrame
            Node data for each side. The first column must be the node key as it appears in data.
        name : string (optional)
            Name of the network. 
            It will be used to export the files.

        Examples
        --------
        >>> 

        NOTE: BiGraph is Undirected!!! This is a BIG problem!!
        """
        self.c = c if c is not None else 'c'
        self.p = p if p is not None else 'p'
        super(mcp,self).__init__(side=self.c,aside=self.p)
        self.data = None
        self.load_links_data(data,use=use)
        self.name = name
        self.size = {self.c:None,self.p:None}
        if nodes_c is not None:
            self.load_nodes_data(self.c,nodes_c)
        if nodes_p is not None:
            self.load_nodes_data(self.p,nodes_p)
        if self.data is not None:
            self.build_net()

    def __str__(self):
        out = ''
        out+= 'Name        : '+self.name +'\n' if self.name != '' else ''
        out+= 'Nodes labels: '+self.c + ' | ' + self.p + ' (' + str(len(self.nodes(self.c))) + 'x' + str(len(self.nodes(self.p))) +')\n'
        out+= 'Ptype       : '+self._ptype +'\n'
        if (self.P[self.side] is not None)|(self.P[self.aside] is not None):
            out+= 'Projections:\n'
            if (self.P[self.side] is not None):
                out+='   '+str(self.P[self.side].node_id)+' ('+str(len(self.P[self.side].nodes()))+')\n'
            if (self.P[self.aside] is not None):
                out+='   '+str(self.P[self.aside].node_id)+' ('+str(len(self.P[self.aside].nodes()))+')\n'
        out = out[:-1]
        return out.encode('utf-8')

    def _check_use(self,use,data):
        if (self.c in data.columns.values)&(use[0] != self.c):
            print 'Warning: Column '+use[0]+' used for nodes type '+self.c
        if (self.p in data.columns.values)&(use[1] != self.p):
            print 'Warning: Column '+use[1]+' used for nodes type '+self.p

    def load_links_data(self,data,use=None):
        """
        Loads the data into the class. 

        Parameters
        ----------
        data : list or DataFrame
            Data on which to build the network.
            If data is a list, it should have the form [(c,p,x),...].
        use  : list
            List of columns to use as nodes ids and weight.
            use has the form [c,p,x].
            If use is not provided, then it will use the first three columns of the DataFrame in the order c,p,x.
        """
        if type(data) != type(DataFrame()):
            data = DataFrame(data)
            use = data.columns.values.tolist()[:3]
        use = data.columns.values[:3].tolist() if use is None else use
        self._check_use(use,data)
        newdata = data[use].rename(columns=dict(zip(use,[self.c,self.p,'x'])))
        if self.data is not None:
            newdata = concat([self.data,newdata])
        newdata = newdata.groupby([self.c,self.p]).sum().reset_index()
        self.data = newdata
        self.add_nodes_from(self.data[self.c].values.tolist(),self.c)
        self.add_nodes_from(self.data[self.p].values.tolist(),self.p)

    def load_nodes_data(self,side,nodes_data,use=None):
        """
        Adds properties to the nodes.

        Parameters
        ----------
        side : int or str
            Tags for each side of the bipartite network.
        nodes_data : pandas DataFrame
            DataFrame with node id in one column and node properties as other columns.
        use : list
            Columns of nodes_data to use.
            The first element of the list has the label of the column with the node id.
            The other elements are the labels of the columns to use.
            If use is not provided, it will look for a column named side.
        """
        self._check_side(side)
        use = [side]+nodes_data.drop(side,1).columns.values.tolist() if use is None else use
        nodes_data = merge(self.nodes(side,as_df=True)[[side]],nodes_data[use],how='left').fillna('NA')[use]
        for name in use[1:]:
            values = dict(zip(nodes_data[use[0]].values,nodes_data[name].values))
            self.set_node_attributes(side,name,values)

    def _calculate_RCA(self):
        self.data = merge(self.data,calculateRCA(self.data,c=self.c,p=self.p,x='x',shares=True).drop('x',1),how='left',left_on=[self.c,self.p],right_on=[self.c,self.p])

    def build_net(self,RCA=True,th=1.,xth=0.):
        '''
        Builds the bipartite network with the given data.
        Warning: If RCA is False then th should be provided.

        Parameters
        ----------
        RCA      : boolean (True)
            Whether to use RCA filtering or regular filtering.
        th       : float (default=1)
            Threshold to use. If RCA=True is the RCA threshold, if RCA=False is the flow threshold.
        '''
        self.P[self.c] = None
        self.P[self.p] = None
        self.remove_edges_from(self.edges())
        header = '' if self.name == '' else self.name + ': '
        if 'RCA' not in self.data.columns.values:
            self._calculate_RCA()
        if RCA:
            net = self.data[(self.data['RCA']>=th)&(self.data['x']>=xth)][[self.c,self.p,'RCA','x']]
        else:
            print 'Warning: th should be provided.'
            net = self.data[self.data['x']>=th][[self.c,self.p,'RCA','x']]
        self.add_edges_from(zip(net[self.c].values,net[self.p].values))
        self.set_edge_attributes('RCA',dict(zip(zip(net[self.c].values,net[self.p].values),net['RCA'].values)))
        self.set_edge_attributes('x',dict(zip(zip(net[self.c].values,net[self.p].values),net['x'].values)))

    def CalculateComplexity(self,th=0.0001):
        '''
        Calculates the Hidalgo-Hausmann Economic Complexity Index.

        Example
        -------
        >> ECI,PCI = M.CalculateComplexity()
        '''
        A = self.edges(as_df=True)
        A['adj']=1
        A = A.pivot(index=self.c,columns=self.p,values='adj').fillna(0)
        ECI,PCI = CalculateComplexity(A.as_matrix(),th=th)
        PCI = DataFrame(zip(A.columns.values,PCI),columns=[self.p,'PCI'])
        ECI = DataFrame(zip(A.index.values,ECI),columns=[self.c,'ECI'])
        return ECI,PCI

    def CalculateComplexityPlus(self,th=0.001,max_iter=5000):
        '''
        Calculates the Hidalgo Economic Complexity Plus Index.

        Example
        -------
        >> ECI,PCI = M.CalculateComplexityPlus()
        '''
        A = self.edges(as_df=True)
        A['adj']=1
        A = A.pivot(index=self.c,columns=self.p,values='adj').fillna(0)
        X = A.as_matrix()
        ECI,PCI = CalculateComplexityPlus(X,th=th,max_iter=max_iter)
        PCI = DataFrame(zip(A.columns.values,PCI),columns=[self.p,'PCIp'])
        ECI = DataFrame(zip(A.index.values,ECI),columns=[self.c,'ECIp'])
        return ECI,PCI

    def filter_nodes(self,side,node_list,keep=True):
        '''
        Filters nodes from the data. 
        Warning: It drops all edges from the bipartite network.

        Parameters
        ----------
        side      : self.c or self.p
            Side to operate on.
        node_list : list
            List of nodes to drop/keep.
        keep      : boolean (True)
            If True it will keep the nodes in node_list and drop everything else. 
            If False it will drop the nodes in node_list and keep everything else.
        '''
        self._check_side(side)
        if keep:
            d = list(set(self.nodes(side)).difference(set(node_list)))
            k = node_list[:]
        else:
            d = node_list[:]
            k = list(set(self.nodes(side)).difference(set(node_list)))
        self.remove_nodes_from(d)
        self.data = merge(DataFrame(k,columns[side]),self.data,how='left')
        if 'RCA' in self.data.columns.values:
            self.data = self.data.drop('RCA',1)
        self.remove_nodes_from(self.edges())

    def avg_inds(self,side):
        """ THIS IS NOT WORKING YET!!!
        Calculates indicators for one side as average over the network of indicators of the other side.
        For each node on side, calculates the average of a given indicator in aside for the bipartite network.
        """
        self._check_side(side)
        if len(self.edges()) is None:
            raise NameError('Empty network found. Please run \n>>> build_net()')


        aside = self.c if side == self.p else self.p
        ns = self.nodes(side,as_df=True)

        inds = []
        for ind in ns.drop(side,1).columns.values:
            try:
                mean(ns[ind].values)
                inds.append(ind)
            except:
                pass
        if len(inds) == 0:
            print self.name + "No indicators to average"
            
        for ind in inds:
            print self.name + 'Calculating average for ',ind
            av_ind = merge(self.net,self.data,how='left',left_on=[side,aside],right_on=[side,aside])[[side,aside,'s_'+side]]
            av_ind = merge(av_ind,self._nodes[side][[side,ind]].dropna(),how='inner',left_on=side,right_on=side)
            av_ind = merge(av_ind,av_ind[[aside,'s_'+side]].groupby(aside).sum().reset_index().rename(columns={'s_'+side:'N_'+aside}),
                how='inner',left_on=aside,right_on=aside)
            av_ind['avg_'+ind] = av_ind['s_'+side]*av_ind[ind]/av_ind['N_'+aside]
            self._nodes[aside] = merge(self._nodes[aside],av_ind[[aside,'avg_'+ind]].groupby(aside).sum().reset_index(),how='left',left_on=aside,right_on=aside)

    def recombination_ease(self,side):
        nodes = self.nodes(side,as_df=True)
        if 'E' not in nodes.columns:
            aside = self.side if side == self.aside else self.aside
            net = self.edges(as_df=True)[[side,aside]]
            dis = merge(net,net,how='inner',left_on=aside,right_on=aside)[[side+'_x',side+'_y']].drop_duplicates()
            dis = dis[dis[side+'_x']!=dis[side+'_y']]
            E = merge(dis.groupby(side+'_x').count().reset_index().rename(columns={side+'_y':'n_c'}).rename(columns={side+'_x':side}),net.groupby(side).count().reset_index().rename(columns={aside:'n_p'}),how='outer').fillna(0)
            E['E'] = E['n_c']/E['n_p']
            E = E[[side,'E']]
            self.set_node_attributes(side,'E',dict(E.values))
            return E
        else:
            return nodes[[side,'E']]

    def NK_complexity(self,side):
        aside = self.c if side == self.p else self.p
        E = self.recombination_ease(aside)

        K = merge(self.edges(as_df=True),E)[[side,aside,'E']]
        K = merge(K.groupby(side).sum()[['E']].reset_index(),K.groupby(side).count()[[aside]].reset_index().rename(columns={aside:'n_c'}))
        K['K'] = K['n_c']/K['E']
        return K[[side,'K']]

    def entropy(self,side):
        data = self.data[[self.c,self.p,'x']]
        S = merge(data,data.groupby(side).sum()[['x']].reset_index().rename(columns={'x':'N_i'}))
        S['s'] = (S['x']/S['N_i'].astype(float))
        S['s'] = -log(S['s'])*S['s']
        S = S.groupby(side).sum()[['s']].reset_index()
        return S

    def _get_size(self,side):
        g = self.projection(side)
        F = {}
        for u in g.nodes():
            w = 0
            for v in g.neighbors(u):
                ww = g.get_edge_data(u,v)
                if ww is not None:
                    w += ww['weight']
            F[u] = w
        self.size[side] = F

    def densities(self,side,m=None,ptype=None):
        '''
        Calculates all the nonzero densities according to the projection of self into side.
        If m is given, it will calculate the densities according to self's projection and m's links.

        For example, self can be the industry-occupation space, and m can be the region-industry space.

        Parameters
        ----------
        side : str
            Side to project M into. 
            To recover the product-space densities, side = p
        m : binet.mcp (optional)
            If provided, it will consider the links between c and p as provided by m.
        ptype : str (default='CP')
            Type of projection to use as weights.

        Returns
        -------
        W : pandas.DataFrame
            Table with the densities:
                aside : country, in the product-space language
                side : product, in the product-space language
                w : density
                mcp : boolean indicating whether side and aside are connected

        '''
        m=self if m is None else m
        aside = self.c if side == self.p else self.p
        maside = m.c if side == m.p else m.p
        if (ptype is not None)&(ptype not in set(['CP','NK'])):
            raise NameError('Projection type not supported:'+str(ptype))
        P = self.projection(side,ptype=ptype)
        P = P.edges(as_df=True)
        P = concat([P,P[[side+'_y',side+'_x','weight']].rename(columns={side+'_y':side+'_x',side+'_x':side+'_y'})]).drop_duplicates().rename(columns={side+'_x':side})
        m = m.edges(as_df=True)[[m.c,m.p]]
        w = merge(m.rename(columns={side:side+'_y'}),P,how='left').fillna(0).groupby([maside,side]).sum()[['weight']].reset_index().rename(columns={'weight':'num'})
        P = P.groupby(side).sum()[['weight']].reset_index().rename(columns={'weight':'den'})
        w = merge(w,P)
        w['w'] = w['num']/w['den']
        w = w[[maside,side,'w']]
        m['mcp'] = 1
        w = merge(w,m,how='outer').fillna(0)
        return w

    def to_csv(self,side=None,th=None,path='',th_low=None,report=False):
        '''Dumps one of the projections into two csv files (name_side_nodes.csv,name_side_edges.csv).
        
        Parameters
        ----------
        side : str
            Side to save the projection. If None it will save the bipartite network.
        th : float
            Threshold to populate the MST with.
        th_low : float (optional)
            Lower threshold to the strength of the links.
        path : str
            Path to save the files to.
        report : boolean (False)
            If True it prints a small summary of the output network.

        Returns
        -------
        _edges.csv : file
            List of edges
        _nodes.csv : file
            List of nodes
        '''
        if (th is None)&(side is not None):
            raise NameError('Must provide a threshold')
        if side is None:
            edges = DataFrame(self.edges(),columns=[self.c,self.p])
            edges.to_csv(path+self.name+'_bipartite_edges.csv',index=False)
            nodes_c = self.nodes(self.c,as_df=True)
            nodes_p = self.nodes(self.p,as_df=True)
            nodes_c.to_csv(path+self.name+'_'+self.c+'_nodes.csv')
            nodes_p.to_csv(path+self.name+'_'+self.p+'_nodes.csv')
            if report:
                print 'Bipartite network'
                print 'Number of edges: ',len(edges)
                print 'Number of nodes '+self.c+': ',len(nodes_c)
                print 'Number of nodes '+self.p+': ',len(nodes_p)
        else:
            dis = DataFrame([(u,v,self.P[side].get_edge_data(u,v)['weight']) for u,v in self.P[side].edges()],columns=[side+'_x',side+'_y','fi'])
            dis = build_connected(dis,th,progress=False)
            dis['fi'] = dis['fi'].astype(float)
            if th_low is None:
                dis.to_csv(path+self.name+'_'+side+'_th'+str(th)+'_edges.csv')
            else:
                dis = dis[dis['fi']>=th_low]
                dis.to_csv(path+self.name+'_'+side+'_th'+str(th)+'_th_low'+str(th_low)+'_edges.csv')
            nns = self.nodes(side,as_df=True)
            if report:
                print 'Upper threshold: ',th
                print 'Lower threshold: ',th_low
                print 'Nodes: ',len(nns)
                print 'Edges: ',len(dis)
            nns.to_csv(path+self.name+'_'+side+'_nodes.csv')






















class tnet(Graph):
    def __init__(self,data,use=None,single_ocurrencies=False,directed=False):
        """
        Class of networks based on the number of ocurrencies of a pair (u,v). 
        Examples are: language network, city mobility network, and labor mobility network.
        Flow network

        Parameters
        ----------
        """
        if directed:
            raise NameError('Directed type is not supported yet')
        self.data = None


        self.load_links_data(data,use=use,directed=directed)


    def load_links_data(self,data,use=None,directed=False):
        """
        Loads the data into the class. 

        Parameters
        ----------
        data : list or DataFrame
            Data on which to build the network.
            If data is a list, it should have the form [(c,p,x),...].
        use  : list
            List of columns to use as nodes ids and weight.
            use has the form [c,p,x].
            If use is not provided, then it will use the first three columns of the DataFrame in the order c,p,x.
        """

        if type(data) != type(DataFrame()):
            data = DataFrame(data)
            use = data.columns.values.tolist()[:3]
        use = data.columns.values[:3].tolist() if use is None else use
        self._check_use(use,data)
        newdata = data[use].rename(columns=dict(zip(use,['s','t','x'])))
        if self.data is not None:
            newdata = concat([self.data,newdata])
        newdata = newdata.groupby(['s','t']).sum().reset_index()
        self.data = newdata
        self.add_nodes_from(self.data[self.c].values.tolist(),self.c)
        self.add_nodes_from(self.data[self.p].values.tolist(),self.p)

    def load_links_data(self,data,u='',v='',n='',single_ocurrencies=False,directed=False):
        """Loads the data into the class.
        """
        self.u = data.columns.values[0] if u=='' else u
        self.v = data.columns.values[1] if v=='' else v
        self.n = data.columns.values[2] if n=='' else n
        self.n1 = 'n1'
        self.n2 = 'n2'
        if single_ocurrencies:
            self.n1 = data.columns.values[3]
            self.n2 = data.columns.values[4]
            self.data = data[[self.u,self.v,self.n,self.n1,self.n2]]
        else:
            self.n1 = None
            self.n2 = None
            self.data = data[[self.u,self.v,self.n]]
        self.data = self.data[self.data[self.u]!=self.data[self.v]]
        self.load_nodes_data()
        if not directed:        
            self.data = merge(self.data,self.nodes,how='left',left_on=self.u,right_on=self.n_id)
            self.data = merge(self.data,self.nodes,how='left',left_on=self.v,right_on=self.n_id)
            invert = (self.data[self.n_id+'_x']<self.data[self.n_id+'_y'])
            invert_u = self.data[invert][self.u]
            invert_v = self.data[invert][self.v]
            self.data.loc[invert,self.u] = invert_v
            self.data.loc[invert,self.v] = invert_u
            self.data = self.data.drop([self.n_id+'_x',self.n_id+'_y'],1)
        self.data = self.data.groupby([self.u,self.v]).sum().reset_index()

    def get_single_ocurrencies(self):
        self.data = merge(self.data,self.data.groupby(self.u).sum()[[self.n]].reset_index().rename(columns={self.n:self.n1})
                ,how='left',left_on=self.u,right_on=self.u)
        self.data = merge(self.data,self.data.groupby(self.v).sum()[[self.n]].reset_index().rename(columns={self.n:self.n2})
                ,how='left',left_on=self.v,right_on=self.v)


    def load_nodes_data(self,n_id='',nodes=None):
        self.n_id = 'node_id' if n_id == '' else n_id
        if self.nodes == None:
            self.nodes = DataFrame(list(set(self.data[self.u].values)|set(self.data[self.v].values)),columns=['node_id']).reset_index().rename(columns={'index':'node_index'})
        self.nodes = self.nodes.rename(columns={'node_id':self.n_id})
        if nodes !=None:
            self.nodes = merge(self.nodes,nodes,how='left',left_on=self.n_id,right_on=self.n_id)
        
    def get_t(self):
        if (self.n1 not in self.data.columns.values)|(self.n2 not in self.data.columns.values):
            self.get_single_ocurrencies()

        N = float(self.data[self.n].sum())
        self.data['D'] = [max(v) for v in zip(data[self.n1].values,data[self.n2].values)]##THIS CAN BE DONE BETTER
        self.data['f'] = (self.data[self.n].astype(float)*N-self.data[self.n1].astype(float)*self.data[self.n2].astype(float))/sqrt(self.data[self.n1].astype(float)*data[self.n2].astype(float)*(N-self.data[self.n1].astype(float))*(N-self.data[self.n2].astype(float)))
        self.data['t'] = self.data['f']*sqrt(self.data['D'].astype(float)-2.)/sqrt(1.-self.data['f']**2)
        self.data = self.data.drop('D',1)

    def t(self):
        return self.data[[self.u,self.v,'t']]


