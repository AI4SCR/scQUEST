# load packages
setwd("~/Documents/projects/scQUEST")
pkgs = readr::read_lines('fcs-requirements.txt')
for(i in pkgs) suppressPackageStartupMessages(library(i, character.only = TRUE))

# define the python interpreter to use
reticulate::use_python('/usr/local/Caskroom/miniconda/base/envs/r-reticulate-env/bin/python')

# helper
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

# files
root = '/Users/art/Library/CloudStorage/Box-Box/STAR_protocol/Revision/Datasets'
root = '/Users/art/Downloads/Datasets'
level = 'Levine_CyTOF_AML_AnnotatedCellTypes'

files = dir(paste(root, level, sep='/'))
files.fcs = files[grepl('.fcs$', files)]

# mappings
rename.sample = readr::read_csv(paste(root, level, 'Levine_CellTypesAnnotated_sampleID.csv', sep='/'))
map.sample = mapper('sample_id', rename.sample)

rename.channel = readr::read_csv(paste(root, level, 'Levine_CellTypesAnnotated_channels.csv', sep='/'))
map.channel = mapper('channel', rename.channel)

# load FCS file
X = NULL
OBS = NULL
VAR = NULL
fcs.header = list()

f.count = 0
for(f in files.fcs){
  f.count = f.count + 1
  cat(f.count, '/', length(files.fcs), ' reading in file ', f, '\n')
  file.name = paste(root, level, f, sep='/')
  
  header = read.FCSheader(file.name)
  header.tbl = tibble(keyword = names(header[[1]]), value=header[[1]])

  fcs <- read.FCS(file.name)
  x = exprs(fcs)
  
  obs = list(fcs_file = f)
  for(i in names(map.sample)){
    obs[[i]] = map.sample[[i]][[f]]
  }
  
  obs = do.call(tibble, obs)
  obs %<>% slice(rep(1, each = dim(x)[1]))
  
  var = parameters(fcs)@data %>% select(-range, -minRange, -maxRange)
  
  X = rbind(X,x)
  OBS = rbind(OBS, obs)
  
  if(is.null(VAR)) VAR = var
  else stopifnot(all(VAR == var))
  
  stopifnot(!f %in% names(fcs.header)) 
  fcs.header[[f]] = header.tbl
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

#saveRDS(ad, paste(root, level, 'ad.rds', sep='/'))
anndata::write_h5ad(ad, paste(root, level, 'ad_annotated_cell_types.h5ad', sep='/'))
