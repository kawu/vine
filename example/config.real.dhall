let
  -- SGD method to use 
  Method =
    < AdaDelta : {decay : Double, eps : Double}
    | Momentum : {gamma : Double, tau : Double, alpha0 : Double}
    | Adam : {alpha0 : Double, tau : Double, beta1 : Double, beta2 : Double, eps : Double}
    >   
in
let
  -- You can optimize for marginal probabilities, or for the (global)
  -- log-likelihood.  The latter leads to simpler and faster training,
  -- but it led to lower results in our VMWE identification experiments.
  ProbTyp = <Marginals : {} | Global : {}> 
in
let
  -- Perform free (unconstrained) or constrained training.  Don't use
  -- the local option, it is not supported at the moment.
  Version = <Free : {} | Constrained : {} | Local : {}>
in
{ sgd =
    -- Number of iterations (epochs) over the entire dataset
  { iterNum = 60
    -- Size of the mini-batch
  , batchSize = 30
    -- Overlap between two subsequent mini-batches.  The value should be 
    -- in the range [0, iterNum).
  , batchOverlap = 15
    -- Randomize the stream of the SGD mini-batches, i.e., shuffle
    -- the training dataset before each epoch.
  , batchRandom = True
    -- Report the value of the objective function every K epochs.
    -- The value of 1.0 is a safe default.
  , reportEvery = 1.0 
  }
  -- Use Adam for optimization (alternatives: AdaDelta, Momentum; see above)
, method = Method.Adam
      -- Initial stepsize
    { alpha0 = 0.001
      -- Time-based stepsize decay parameter.  Represents the number of 
      -- iterations after which the initial stepsize is halved.
    , tau = 15.0
      -- The exponential decay rates
    , beta1 = 0.9
    , beta2 = 0.999
      -- Epsilon
    , eps = 1.0e-8
    }
  -- Objective configuration
, probCfg =
      -- Use marginal probabilities
    { probTyp = ProbTyp.Marginals {=} 
      -- Do not use constrained training
    , version = Version.Free {=} 
    }   
}
