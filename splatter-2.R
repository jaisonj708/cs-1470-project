suppressMessages(library(splatter))

# # Create mock data
# library(scater)
# set.seed(1)
# sce <- mockSCE()

nGroups<- 2
distr  <- rep(1, nGroups) / nGroups
method <- 'groups'

nGenes   <- 200 #1000
nCells   <- 2000 #10000
typeDP   <- 'experiment'
shapeDP  <- -1
midDP    <- 5

sim2 <- suppressMessages(splatSimulate(group.prob=distr,nGenes=nGenes,batchCells=nCells,
                                       dropout.type=typeDP,method=method,
                                       dropout.shape=shapeDP,dropout.mid=midDP,seed=42))

counts     <- as.data.frame(t(counts(sim2)))
truecounts <- as.data.frame(t(assays(sim2)$TrueCounts))

dropout    <- assays(sim2)$Dropout
mode(dropout) <- 'integer'
dropout    <- as.data.frame(t(dropout))

cellinfo   <- as.data.frame(colData(sim2))
geneinfo   <- as.data.frame(rowData(sim2))

