# load packages
pkgs = c('flowCore',
         'anndata',
         'tidyverse',
         'magrittr',
         'httr')
for(i in pkgs) suppressPackageStartupMessages(library(i, character.only = TRUE))

# define the python interpreter to use
reticulate::use_python('/usr/local/Caskroom/miniconda/base/envs/r-reticulate-env/bin/python')

# request for figshare api
project_id = 140062
articel_id = 19867021
PATH = '~/.scQUEST/Levine_CyTOF_AML_AnnotatedCellTypes/'
REQUEST = paste0('https://api.figshare.com/v2/articles/', articel_id, '/files')

# helpers
mapper = function(key, df){
  mapping= list()
  cols = colnames(df)
  cols = cols[cols != key]
  for(i in cols){
    m = map(df[[i]], ~ .x)
    names(m) = df[[key]]
    mapping[[i]] = m
  }
  return (mapping)
}

download_files = function(path = PATH, request = REQUEST){
  path = path.expand(path)
  dir.create(path, showWarnings = TRUE, recursive = TRUE, mode = "0777")
  
  resp = GET(request)
  stop_for_status(resp)
  
  files = content(resp, 'parsed')
  files = map(files, function(x){
    url = x$download_url
    fpath = paste0(path, x$name)
    if(!file.exists(fpath)){
      download.file(url, fpath)
    }
    return(c(url, fpath))
  })
}

# download files
files = download_files(PATH, REQUEST)
files = unlist(map(files, ~.x[2]))
files.fcs = files[grepl('.fcs$', files)]
files.csv = files[grepl('.csv$', files)]

files.sampleID = files.csv[grepl('sampleID', files.csv)]
files.channels = files.csv[grepl('channels', files.csv)]

# mappings
rename.sample = readr::read_csv(files.sampleID)
rename.sample  %<>% rename(sample_id = sample_ID)
map.sample = mapper('sample_id', rename.sample)

rename.channel = readr::read_csv(files.channels)
map.channel = mapper('channel', rename.channel)

# load FCS file
X = NULL
OBS = NULL
VAR = NULL
fcs.header = list()

f.count = 0
for(f in files.fcs){
  f.count = f.count + 1
  f.base =  basename(f)
  cat(f.count, '/', length(files.fcs), ' reading in file ', f.base, '\n')
  
  header = read.FCSheader(f)
  header.tbl = tibble(keyword = names(header[[1]]), value=header[[1]])
  
  fcs <- read.FCS(f)
  x = exprs(fcs)
  
  obs = list(fcs_file = f.base)
  for(i in names(map.sample)){
    obs[[i]] = map.sample[[i]][[f.base]]
  }
  
  obs = do.call(tibble, obs)
  obs %<>% slice(rep(1, each = dim(x)[1]))
  
  var = parameters(fcs)@data %>% select(-range, -minRange, -maxRange)
  
  X = rbind(X,x)
  OBS = rbind(OBS, obs)
  
  if(is.null(VAR)) VAR = var
  else stopifnot(all(VAR == var))
  
  stopifnot(!f %in% names(fcs.header)) 
  fcs.header[[f.base]] = header.tbl
}

# tidy data
VAR %<>% rename(channel = name, marker = desc)
for(i in names(map.channel)){
  VAR %<>% mutate("{i}" := unlist(map.channel[[i]][channel])) 
}


# create AnnData object
ad = AnnData(
  X = X,
  var = VAR,
  obs = OBS,
  uns = list(
    fcs_header = fcs.header
  )
)

path.output = path.expand(paste0(PATH, 'ad_healty_aml.h5ad'))
anndata::write_h5ad(ad, path.output)
