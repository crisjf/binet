from pandas import DataFrame,merge
from numpy import sqrt,mean,corrcoef
from networkx import Graph
from functions import calculateRCA,CalculateComplexity,build_connected,build_html
from functions_gt import get_pos
from os import getcwd
import webbrowser
from copy import deepcopy



class BiGraph(Graph):
    def __init__(self,side=0,aside=1):
        """Wrapper class for networkx Graph to deal with bipartite networks. 
        It adds a property called 'side' to each node.
        All the other functions work the same way."""
        super(BiGraph,self).__init__()
        self.side  = side
        self.aside = aside
    
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
    
    def nodes(self,side):
        if side not in [self.side,self.aside]:
            raise NameError('Wrong side label choose between '+str(self.side)+' and '+str(self.aside))
        return [u for u in self.nodes_iter() if self.node[u]['side'] == side]



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
        if side not in [self.c,self.p]:
            raise NameError('Wrong label, choose between '+self.c+' and '+self.p)
        return self._nodes[side].drop(side+'_index',1)

    def edges(self):
        if self.net is None:
            raise NameError('No network defined. Please run \n>>> build_net()')
        return self.net[[self.c,self.p]]

    def avg_inds(self,side):
        if side not in [self.c,self.p]:
            raise NameError('Wrong label, choose between '+self.c+' and '+self.p)
        """For each node on side, calculates the average of a given indicator in aside for the bipartite network"""
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
        if side not in [self.c,self.p]:
            raise NameError('Wrong label, choose between '+self.c+' and '+self.p)
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
        if trimmed|as_network:
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
            return self.projection_d[side]


    def to_csv(self,side,path='',trimmed=False):
        '''Dumps one of the projections into two csv files (name_side_nodes.csv,name_side_edges.csv).'''
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
        if side not in [self.c,self.p]:
            raise NameError('Wrong label, choose between '+self.c+' and '+self.p)
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