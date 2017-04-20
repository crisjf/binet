class mcp_old(object):
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


    def _project(self,side):
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
                self._project(side)
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
