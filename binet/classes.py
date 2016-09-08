from pandas import DataFrame,merge
from numpy import sqrt,mean,corrcoef
from networkx import Graph,set_node_attributes,set_edge_attributes
from functions import calculateRCA,CalculateComplexity,build_connected,build_html
from functions_gt import get_pos
from os import getcwd
import webbrowser
from copy import deepcopy
from itertools import chain 



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
        '''Will only return the properties when data is asked as a dataframe.'''
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
        if not as_df:
            return super(BiGraph,self).edges(nbunch=None, data=False, default=None)
        else:
            es = list(super(BiGraph,self).edges(nbunch=None, data=False, default=None))
            return DataFrame(es,columns=[self.side,self.aside])

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

    def project(self,side):
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
        self.P[side].add_weighted_edges_from([val[1:] for val in dis.itertuples()])


    def projection(self,side):
        """
        Returns the network projection (will calculate if it does not exist).

        Parameters
        ----------
        side : int or str
            Tags for each side of the bipartite network.

        Returns
        ----------
        P : networkx.Graph() object
            Weighted betwork containing the projected network.
        """
        if self.P[side] is None:
            self.project(side)
        return self.P[side]



class mcp_new(BiGraph):
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
        """
        super(mcp_new,self).__init__(side=c,aside=p)
        self.c = c if c is not None else 'c'
        self.p = p if p is not None else 'p'
        self.side,self.aside  = self.c,self.p
        self.data = None
        self.load_links_data(data,use=use)
        self.name = name
        if nodes_c is not None:
            self.load_nodes_data(self.c,nodes_c)
        if nodes_p is not None:
            self.load_nodes_data(self.p,nodes_p)

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

    def build_net(self,RCA=True,th=1.):
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
        self.remove_nodes_from(self.edges())
        header = '' if self.name == '' else self.name + ': '
        if 'RCA' not in self.data.columns.values:
            self._calculate_RCA()
        if RCA:
            net = self.data[self.data['RCA']>=th][[self.c,self.p,'RCA','x']]
        else:
            print 'Warning: th should be provided.'
            net = self.data[self.data['x']>=th][[self.c,self.p,'RCA','x']]
        self.add_edges_from(zip(net[self.c].values,net[self.p].values))
        self.set_edge_attributes('RCA',dict(zip(zip(net[self.c].values,net[self.p].values),net['RCA'].values)))
        self.set_edge_attributes('x',dict(zip(zip(net[self.c].values,net[self.p].values),net['x'].values)))

        

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



class mcp(object):
    def __init__(self,data,name='',nodes_c=None,nodes_p=None,c='',p='',x=''):
        """
        data can be a dataframe with three columns, or a list of tuples
        c,p,x are the labels of the columns to use in the dataframe
        name is used to export to files
        """
        self.name   = name
        self.net    = None
        self.G      = None
        self.data   = None
        self._nodes = None

        self.M = None #Adjacency matrix placeholder
        self.load_links_data(data=data,c=c,p=p,x=x)
        self.load_nodes_data(nodes_c=nodes_c,nodes_p=nodes_p)
        self.projection_d = {self.c:None,self.p:None}
        self.projection_t = {self.c:None,self.p:None}
        self.projection_th = {self.c:None,self.p:None}

        self.size = {self.c:None,self.p:None}
        
    def load_links_data(self,data,c='',p='',x=''):
        """Loads the data into the class.
        """
        if type(data) == type(DataFrame()):
            self.c = data.columns.values[0] if c=='' else c
            self.p = data.columns.values[1] if p=='' else p
            self.x = data.columns.values[2] if x=='' else x
            self.data = data[[self.c,self.p,self.x]]
        else:
            self.c = 'c' if c=='' else c
            self.p = 'p' if p=='' else p
            self.x = 'x' if x=='' else x
            self.data = DataFrame(data).rename(columns={0:self.c, 1: self.p, 2: self.x})
        self.data = self.data.groupby([self.c,self.p]).sum().reset_index()
              
    def load_nodes_data(self,nodes_c=None,nodes_p=None):
        """Adds node properties to the nodes."""
        if self._nodes is None:
            self._nodes = {}
            self._nodes[self.c] = self.data[[self.c]].drop_duplicates().reset_index().drop('index',1).reset_index().rename(columns={'index':self.c+'_index'})
            self._nodes[self.p] = self.data[[self.p]].drop_duplicates().reset_index().drop('index',1).reset_index().rename(columns={'index':self.p+'_index'})
        if nodes_c is not None:
            cols = set(nodes_c.columns.values)
            if self.c not in cols:
                raise NameError('Column '+self.c+' must be included in the node data.')
            self._nodes[self.c] = merge(self._nodes[self.c],nodes_c,how='left',left_on=self.c,right_on=self.c).drop_duplicates()
        if nodes_p is not None:
            cols = set(nodes_p.columns.values)
            if self.p not in cols:
                raise NameError('Column '+self.p+' must be included in the node data.')
            self._nodes[self.p] = merge(self._nodes[self.p],nodes_p,how='left',left_on=self.p,right_on=self.p).drop_duplicates()

    def filter_nodes(self,nodes_c=None,nodes_p=None):
        '''Takes a list of nodes or a DataFrame with one columns as nodes, and filters out these nodes from the data.
        If any projection was calculated, it will not be altered, therefore it should be recalculated.
        This function only alters self.nodes and self.data'''
        if (nodes_c is None)&(nodes_p is None):
            raise NameError("Must provide list of nodes")
        if nodes_c is not None:
            nodes_c = DataFrame(list(set(nodes_c)),columns=[self.c]) if type(nodes_c) == type([]) else nodes_c[[self.c]].drop_duplicates()
            self._nodes[self.c] = merge(nodes_c,self._nodes[self.c],how='left',left_on=self.c,right_on=self.c)
            self.data = merge(nodes_c,self.data,how='left',left_on=self.c,right_on=self.c)
        if nodes_p is not None:
            nodes_p = DataFrame(list(set(nodes_p)),columns=[self.p]) if type(nodes_p) == type([]) else nodes_p[[self.p]].drop_duplicates()
            self._nodes[self.p] = merge(nodes_p,self._nodes[self.p],how='left',left_on=self.p,right_on=self.p)
            self.data = merge(nodes_p,self.data,how='left',left_on=self.p,right_on=self.p)
        if 'RCA' in self.data.columns.values:
            self.data = self.data.drop('RCA',1)
        self.net = None

    def _calculate_RCA(self):
        self.data = merge(self.data,calculateRCA(self.data,c=self.c,p=self.p,x=self.x,shares=True).drop(self.x,1),how='left',left_on=[self.c,self.p],right_on=[self.c,self.p])

    def build_net(self,RCA=True,th = 1.,progress=True):
        '''If RCA is set to False then th should be provided.
        Builds the bipartite network with the given data.'''
        if (self.name!='')&progress:
            print self.name + ': '+'Building bipartite network\nRCA = '+str(RCA)+'\nth = '+str(th)
        elif progress:
            print 'Building bipartite network\nRCA = '+str(RCA)+'\nth = '+str(th)
        if RCA:
            if 'RCA' not in self.data.columns.values:
                self._calculate_RCA()
            self.net = self.data[self.data['RCA']>=th][[self.c,self.p]]
        else:
            print 'Warning: th should be provided.'
            self.net = self.data[self.data[self.x]>=th][[self.c,self.p]]
        nc = len(set(self.net[self.c].values).intersection(set(self._nodes[self.c][self.c].values)))
        np = len(set(self.net[self.p].values).intersection(set(self._nodes[self.p][self.p].values)))
        if progress:
            print 'N nodes_c = '+str(nc)
            if nc != len(self._nodes[self.c]):
                print '\t('+str(len(self._nodes[self.c])-nc)+' nodes were dropped)\n'
            print 'N nodes_p = '+str(np)
            if np != len(self._nodes[self.p]):
                print '\t('+str(len(self._nodes[self.p])-np)+' nodes were dropped)\n'
            print 'N edges = '+str(len(self.net))

        self.G = BiGraph(side=self.c,aside=self.p)
        self.G.add_edges_from(zip(self.net[self.c].values,self.net[self.p].values))


    def nodes(self,side):
        self._check_side(side)
        return self._nodes[side].drop(side+'_index',1)

    def edges(self):
        if self.net is None:
            raise NameError('No network defined. Please run \n>>> build_net()')
        return self.net[[self.c,self.p]]

    def avg_inds(self,side):
        """Calculates indicators for one side as average over the network of indicators of the other side.
        For each node on side, calculates the average of a given indicator in aside for the bipartite network.
        """
        self._check_side(side)
        if self.net is None:
            raise NameError('No network defined. Please run \n>>> build_net()')
        aside = self.c if side == self.p else self.p
        inds = []
        for ind in self._nodes[side].drop([side+'_index',side],1).columns.values:
            try:
                mean(self._nodes[side][ind].values)
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


    def project(self,side):
        """Builds the projection of the bipartite network on to the chosen side."""
        self._check_side(side)
        if self.net is None:
            self.build_net()
        
        aside = self.c if side == self.p else self.p
        dis = merge(self.net,self.net,how='inner',left_on=aside,right_on=aside).groupby([side+'_x',side+'_y']).count().reset_index()
        dis = merge(dis,self._nodes[side],how='left',right_on=side,left_on=side+'_x').drop(side,1)
        dis = merge(dis,self._nodes[side],how='left',right_on=side,left_on=side+'_y').drop(side,1)
        dis = dis[dis[side+'_index_x']>dis[side+'_index_y']].rename(columns={aside:'n_both'}).drop([side+'_index_x',side+'_index_y'],1)
        nside = self.net.groupby(side).count().reset_index().rename(columns={aside:'n_'})
        dis = merge(dis,nside,
                how='inner',left_on=side+'_x',right_on=side).rename(columns={'n_':'n_x'}).drop(side,1)
        dis = merge(dis,nside,
                how='inner',left_on=side+'_y',right_on=side).rename(columns={'n_':'n_y'}).drop(side,1)
        dis['p_x'] = dis['n_both']/dis['n_x'].astype(float)
        dis['p_y'] = dis['n_both']/dis['n_y'].astype(float)
        dis['fi'] = dis[['p_x','p_y']].min(1)
        dis = dis[[side+'_x',side+'_y','fi']]
        self.projection_d[side] = dis
    
    def projection(self,side,progress=True,trimmed=False,as_network=False):
        """Used to access the projection of the bipartite network"""
        if trimmed:
            if self.projection_t[side] is None:
                raise NameError('Projection not trimmed, please run \n>>> M.trim_projection(side,th)')
            if as_network:
                P = self.projection_t[side][[side+'_x',side+'_y','fi']]
                g = Graph()
                g.add_edges_from(zip(P[side+'_x'].values.tolist(),P[side+'_y'].values.tolist(),[{'weight':w} for w in P['fi'].values.tolist()]))
                return g
            else:
                return self.projection_t[side]
        else:
            if self.projection_d[side] is None:
                if progress:
                    print self.name +': ' + 'Calculating projection on '+str(side)
                self.project(side)
            if as_network:
                P = self.projection_d[side][[side+'_x',side+'_y','fi']]
                g = Graph()
                g.add_edges_from(zip(P[side+'_x'].values.tolist(),P[side+'_y'].values.tolist(),[{'weight':w} for w in P['fi'].values.tolist()]))
                return g
            else:
                return self.projection_d[side]


    def to_csv(self,side=None,path='',trimmed=False):
        '''Dumps one of the projections into two csv files (name_side_nodes.csv,name_side_edges.csv).'''
        if side is None:
            self.net.to_csv(path+self.name+'_bipartite_edges.csv',index=False)
        else:
            if self.projection_d[side] is None:
                raise NameError('No projection available. Please run \n>>> M.projection(side)')
            self.nodes(side).to_csv(path+self.name+'_'+side+'_nodes.csv')
            if trimmed:
                if self.projection_t[side] is None:
                    raise NameError('Projection not trimmed. Please run \n>>> M.trim_projection(side,th)')
                self.projection(side,trimmed=True).to_csv(path+self.name+'_'+side+'_th'+str(self.projection_th[side])+'_edges.csv')
            else:
                self.projection(side).to_csv(path+self.name+'_'+side+'_edges.csv')

    def _as_matrix(self):
        '''Returns the network as an adjacency matrix. Useful to calculate complexity.'''
        if self.net is None:
            raise NameError('No network available. Please run \n>>> build_net()')
        self.net['value'] = 1
        dfM = self.net.pivot(index=self.c,columns=self.p,values='value').fillna(0)
        self.net = self.net.drop('value',1)
        self.M = dfM.as_matrix()
        return self.M,dfM.index.values.tolist(),dfM.columns.values.tolist()

    def CalculateComplexity(self):
        if (self.c+'_CI' in self._nodes[self.c].columns.values)|(self.p+'_CI' in self._nodes[self.p].columns.values):
            self._nodes[self.c] = self._nodes[self.c].drop(self.c+'_CI',1)
            self._nodes[self.p] = self._nodes[self.p].drop(self.p+'_CI',1)
        M,cs,ps = self._as_matrix()
        ECI,PCI = CalculateComplexity(M)
        ECI = DataFrame(ECI,columns=[self.c+'_CI'],index=cs).reset_index().rename(columns={'index':self.c})
        PCI = DataFrame(PCI,columns=[self.p+'_CI'],index=ps).reset_index().rename(columns={'index':self.p})
        self._nodes[self.c] = merge(self._nodes[self.c],ECI,how = 'left',left_on=self.c,right_on=self.c)
        self._nodes[self.p] = merge(self._nodes[self.p],PCI,how = 'left',left_on=self.p,right_on=self.p)
        return ECI,PCI

    def trim_projection(self,side,th):
        self._check_side(side)
        self.projection_th[side] = th
        self.projection_t[side] = build_connected(self.projection(side,progress=False),th,progress=False)
        if 'x' in self._nodes[side].columns.values:
            self._nodes[side] = self._nodes[side].drop(['x','y'],1)


    def _get_projection_pos(self,side,C=None):
        '''This function requires graph_tool'''
        if (self.projection_t[side]is None):
            raise NameError('Please run trim_projection(side,th) first')

        pos = get_pos(self.projection_t[side][[side+'_x',side+'_y']],node_id=side,comms=True,progress=False,C=C)
        
        if 'x' in self._nodes[side].columns.values:
            self._nodes[side] = self._nodes[side].drop(['x','y'],1)
        if 'c' in self._nodes[side].columns.values:
            self._nodes[side] = self._nodes[side].drop('c',1)
        
        self._nodes[side] = merge(self._nodes[side],pos)


    def draw_projection(self,side,C=None,path='',show=True,color=True,max_colors=15):
        '''path must be the absolut path, including the "/" symbol at the end. 
        This function requires graph_tool'''
        if (self.projection_t[side]is None):
            raise NameError( 'Please run \n>>> M.trim_projection(side,th)')

        cs = self.projection_t[side][[side+'_x',side+'_y']]
        
        if 'x' not in self._nodes[side].columns.values:
            self._get_projection_pos(side,C=C)

        path = 'file://'+getcwd()+'/' if path == '' else path+'/'

        props = [val for val in self._nodes[side].columns.values.tolist() if val not in set([side,side+'_index','x','y'])]
        if color:
            html = build_html(self._nodes[side],cs,node_id=side,source_id=side+'_x',target_id=side+'_y',color='c',props=props,progress=False,max_colors=max_colors)
        else:
            html = build_html(self._nodes[side],cs,node_id=side,source_id=side+'_x',target_id=side+'_y',props=props,progress=False)
        out = side+'_'+self.name+'_th'+str(self.projection_th[side])+'.html' 
        open(out,mode='w').write(html.encode('utf-8'))
        if show:
            webbrowser.open(path+out)
            print 'OUT: ',path+out

    def _get_size(self,side):
        g = self.projection(side,as_network=True)
        F = {}
        for u in g.nodes():
            w = 0
            for v in g.neighbors(u):
                ww = g.get_edge_data(u,v)
                if ww is not None:
                    w += ww['weight']
            F[u] = w
        self.size[side] = F

    def _check_side(self,side):
        if side not in [self.c,self.p]:
            raise NameError('Wrong side label, choose between '+self.c+' and '+self.p)

    def densities(self,side,m,progress=True):
        '''m must be another object of type mcp'''
        self._check_side(side)
        if self.size[side] is None:
            self._get_size(side)
        F = self.size[side]
        aside = self.c if side == self.p else self.p
        g = self.projection(side,as_network=True)

        if progress:
            print 'Calculating densities based on ' + side +' space'

        W = []
        for c in self.G.nodes(aside):
            f = {}
            ps = set(m.G.neighbors(c))
            for u in g.nodes():
                w = 0.
                for v in ps:
                    ww = g.get_edge_data(u,v)
                    if ww is not None:
                        w += ww['weight']
                f[u] = w
            W += [(c,u,f[u]/F[u],(u in ps)) for u in g.nodes()]
        return DataFrame(W,columns=[aside,side,'w','mcp'])




class tnet(object):
    def __init__(self,data,u='',v='',n='',single_ocurrencies=False,directed=False):
        """
        Class of networks based on the number of ocurrencies of a pair (u,v). 
        Examples are: language network, city mobility network, and labor mobility network.
        """
        if directed:
            raise NameError('Directed type is not supported yet')
        self.load_links_data(data,u=u,v=v,n=n,single_ocurrencies=single_ocurrencies,directed=directed)

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



class compare_nets():
    def __init__(self,M1_,M2_):
        if (M1_.name!='')|(M2_.name!=''):
            self.n1 = 'Unnamed' if M1_.name == '' else M1_.name
            self.n2 = 'Unnamed' if M2_.name == '' else M2_.name

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
        self.c_total_nodes     = len(nodes1|nodes2)
        self.c_repeated_nodes  = len(nodes1.intersection(nodes2))
        self.c_f_missing_nodes = 1.-len(nodes1.intersection(nodes2))/float(len(nodes1|nodes2))
        self.c_f_missing_1     = 1.-len(nodes1)/float(len(nodes1|nodes2))
        self.c_f_missing_2     = 1.-len(nodes2)/float(len(nodes1|nodes2))

        nodes_c = list(nodes1.intersection(nodes2))
        nodes = DataFrame(nodes_c,columns=['node_id'])
        edges1 = merge(edges1,nodes,how='right',left_on=M1.c,right_on='node_id').drop('node_id',1)
        edges2 = merge(edges2,nodes,how='right',left_on=M2.c,right_on='node_id').drop('node_id',1)

        nodes1 = set(M1.nodes(M1.p)[M1.p].values)
        nodes2 = set(M2.nodes(M2.p)[M2.p].values)
        self.p_total_nodes     = len(nodes1|nodes2)
        self.p_repeated_nodes  = len(nodes1.intersection(nodes2))
        self.p_f_missing_nodes = 1.-len(nodes1.intersection(nodes2))/float(len(nodes1|nodes2))
        self.p_f_missing_1     = 1.-len(nodes1)/float(len(nodes1|nodes2))
        self.p_f_missing_2     = 1.-len(nodes2)/float(len(nodes1|nodes2))
            

        nodes_p = list(nodes1.intersection(nodes2))
        nodes = DataFrame(nodes_p,columns=['node_id'])
        edges1 = merge(edges1,nodes,how='right',left_on=M1.p,right_on='node_id').drop('node_id',1)
        edges2 = merge(edges2,nodes,how='right',left_on=M2.p,right_on='node_id').drop('node_id',1)
    
        edges1['edge'] = edges1[M1.c].astype(str)+'-'+edges1[M1.p].astype(str)
        edges2['edge'] = edges2[M2.c].astype(str)+'-'+edges2[M1.p].astype(str)
        edges1 = set(edges1['edge'].values.tolist())
        edges2 = set(edges2['edge'].values.tolist())

        self.e_total_edges     = len(edges1|edges2)
        self.e_f_missing_edges = 1.-len(edges1.intersection(edges1))/float(len(edges1|edges2))
        self.e_f_missing_1     = 1.-len(edges1)/float(len(edges1|edges2))
        self.e_f_missing_2     = 1.-len(edges2)/float(len(edges1|edges2))

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

        self.f_total_edges     = len(edges1|edges2)
        self.f_f_missing_edges = 1.-len(edges1.intersection(edges1))/float(len(edges1|edges2))
        self.f_f_missing_1     = 1.-len(edges1)/float(len(edges1|edges2))
        self.f_f_missing_2     = 1.-len(edges2)/float(len(edges1|edges2))


        cp = merge(M1.projection(M1.c).rename(columns={'fi':'fi_1'}),M2.projection(M2.c).rename(columns={'fi':'fi_2'}),how='inner')
        self.proj_c = corrcoef(cp['fi_1'],cp['fi_2'])[0,1]
        cp = merge(M1.projection(M1.p).rename(columns={'fi':'fi_1'}),M2.projection(M2.p).rename(columns={'fi':'fi_2'}),how='inner')
        self.proj_p = corrcoef(cp['fi_1'],cp['fi_2'])[0,1]


        self.c_m1 = M1.c
        self.c_m2 = M2.c
        self.p_m1 = M1.p
        self.p_m2 = M2.p

    def summary(self):
        print 'Summary of comparing networks '+self.n1+' with '+self.n2
        print ','.join(list(set([self.c_m1,self.c_m2])))
        print '\tTotal nodes          :',self.c_total_nodes
        print '\tTotal repeated nodes :',self.c_repeated_nodes
        print '\tFracion missing nodes:',self.c_f_missing_nodes
        print '\tMissing nodes on 1   :',self.c_f_missing_1
        print '\tMissing nodes on 2   :',self.c_f_missing_2
        print ''
        print ','.join(list(set([self.p_m1,self.p_m2])))
        print '\tTotal nodes          :',self.p_total_nodes
        print '\tTotal repeated nodes :',self.p_repeated_nodes
        print '\tFracion missing nodes:',self.p_f_missing_nodes
        print '\tMissing nodes on 1   :',self.p_f_missing_1
        print '\tMissing nodes on 2   :',self.p_f_missing_2
        print ''
        print ','.join(list(set([str(self.c_m1)+'-'+str(self.p_m1),str(self.c_m2)+'-'+str(self.p_m2)])))
        print '\tTotal edges          :',self.e_total_edges
        print '\tFracion missing edges:',self.e_f_missing_edges
        print '\tMissing edges on 1   :',self.e_f_missing_1
        print '\tMissing edges on 2   :',self.e_f_missing_2
        print ''
        print 'Filtered '+','.join(list(set([str(self.c_m1)+'-'+str(self.p_m1),str(self.c_m2)+'-'+str(self.p_m2)])))
        print '\tTotal edges          :',self.f_total_edges
        print '\tFracion missing edges:',self.f_f_missing_edges
        print '\tMissing edges on 1   :',self.f_f_missing_1
        print '\tMissing edges on 2   :',self.f_f_missing_2
        print ''
        print 'Projection ',','.join(list(set([self.c_m1,self.c_m2])))
        print 'Correlation between fis:',self.proj_c
        print ''
        print 'Projection ',','.join(list(set([self.p_m1,self.p_m2])))
        print 'Correlation between fis:',self.proj_p
        print ''