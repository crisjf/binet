from requests import get
import io,urllib2,bz2

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
