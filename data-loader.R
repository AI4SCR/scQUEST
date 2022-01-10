
# load packages
pkgs = readr::read_lines('fcs-requirements.txt')
for(i in pkgs) suppressPackageStartupMessages(library(i, character.only = TRUE))

# helper
metaFromFileName = function(f){
  
  # patterns
  regex_tissue_type = '^[HNT]?'
  regex_patient_number = 'BB[0-9]{3}'
  regex_breast = '.*BB[0-9]{3}([LR]?).*'
  regex_tumor_region = '.*BB[0-9]{3}[LR]?([ab])?.*'
  regex_plate = 'Plate[1-2]?'
  regex_plate.location = '[12]-[B-F][0-9]{1,2}'
  regex_gadolinium = '.*_GD(neg|pos).fcs$'
  
  # extract
  tissue_type = regmatches(f, regexpr(regex_tissue_type, f))
  patient_number = regmatches(f, regexpr(regex_patient_number, f))
  breast = sub(regex_breast, '\\1', f)
  tumor_region = sub(regex_tumor_region, '\\1', f)
  plate = regmatches(f, regexpr(regex_plate, f))
  plate.location = regmatches(f, regexpr(regex_plate.location, f))
  gadolinium = sub(regex_gadolinium, '\\1', f)
  
  # verify extraction
  stopifnot(
    nchar(tissue_type) == 1,
    nchar(patient_number) == 5,
    nchar(breast) == 0 | nchar(breast) == 1,
    nchar(tumor_region) == 0 | nchar(tumor_region) == 1,
    nchar(plate) == 6,
    nchar(plate.location) >= 4 & nchar(plate.location) <= 5,
    nchar(gadolinium) == 3
    )
  
  f.meta = tibble(tissue_type = tissue_type,
                  patient_number = patient_number, 
                  breast = breast,
                  tumor_region = tumor_region,
                  plate = plate,
                  plate_location = plate.location,
                  gadolinium_status = gadolinium,
                  fcs_file = f
  )
  
  return(f.meta)
}

# files
root = '/Users/art/Library/CloudStorage/Box-Box/STAR_protocol/Data/'
level = 'Cells_CisPtneg_GDneg'

files = dir(paste(root, level, sep='/'))
files.fcs = files[grepl('.fcs$', files)]

# mappings
rename.sample = readr::read_csv(paste(root, level, 'Sample_rename.csv', sep='/'))
map.sample = map(rename.sample$sample_rename, ~ .x)
names(map.sample) = rename.sample$sample

rename.channel = readr::read_csv(paste(root, level, 'Channel_rename.csv', sep='/'))
map.channels = map(rename.channel$channel_rename, ~ .x)
names(map.channels) = rename.channel$channel

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
  
  obs = metaFromFileName(f)[rep(1, nrow(fcs)), ]
  var = parameters(fcs)@data %>% select(-range, -minRange, -maxRange)
  x = exprs(fcs)
  
  X = rbind(X,x)
  OBS = rbind(OBS, obs)
  
  if(is.null(VAR)) VAR = var
  else stopifnot(all(VAR == var))
  
  stopifnot(!f %in% names(fcs.header)) 
  fcs.header[[f]] = header.tbl
}

# tidy data
VAR %<>% rename(channel = name) %>%
  mutate(desc = unlist(map.channels[channel]))

var2obs = c('Center', 'EventLength', 'Offset', 'Residual', 'Time', 'Width', 'beadDist')
obs = X[, VAR$desc %in% var2obs] %>% as_tibble()
names(obs) = VAR$desc[VAR$desc %in% var2obs]
OBS %<>% cbind(obs)

X = X[, !VAR$desc %in% var2obs]
VAR %<>% filter(!VAR$desc %in% var2obs)

# create AnnData object
ad = AnnData(
  X = X,
  var = VAR,
  obs = OBS,
  uns = list(
    fcs_header = fcs.header
  )
)

# range_max = apply(ad$X, 1, function(x) max(x))
# range_min = apply(ad$X, 1, function(x) min(x))
# ad$obs %<>% mutate(range_min = range_min, range_max = range_max)
# 
# range_max = apply(ad$X, 2, function(x) max(x))
# range_min = apply(ad$X, 2, function(x) min(x))
# ad$var %<>% mutate(range_min = range_min, range_max = range_max)

saveRDS(ad, paste(root, level, 'ad.rds', sep='/'))
anndata::write_h5ad(ad, paste(root, level, 'ad.h5ad', sep='/'))
