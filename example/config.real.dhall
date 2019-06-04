let
  Method =
    < AdaDelta : {decay : Double, eps : Double}
    | Momentum : {gamma : Double, tau : Double, alpha0 : Double}
    | Adam : {alpha0 : Double, tau : Double, beta1 : Double, beta2 : Double, eps : Double}
    >   
in
let
  ProbTyp = <Marginals : {} | Global : {}> 
in
let
  Version = <Free : {} | Constrained : {} | Local : {}>
in
{ sgd =
  { iterNum = 60
  , batchSize = 30
  , batchOverlap = 15
  , batchRandom = True
  , reportEvery = 1.0 
  }
, method = Method.Adam
    { alpha0 = 0.001
    , tau = 15.0
    , beta1 = 0.9 
    , beta2 = 0.999
    , eps = 1.0e-8
    }   
, probCfg =
    { probTyp = ProbTyp.Marginals {=} 
    , version = Version.Free {=} 
    }   
}
