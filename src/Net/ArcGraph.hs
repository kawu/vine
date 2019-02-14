{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- {-# LANGUAGE TypeApplications    #-}

-- To derive Binary for `Param`
{-# LANGUAGE DeriveAnyClass #-}

-- To make GHC automatically infer that `KnownNat d => KnownNat (d + d)`
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}


module Net.ArcGraph
  ( 
  -- * Network
    Param(..)
  , size
  , new
  , Graph(..)
  , Arc
  , run
  , eval
  , runTest

  -- * Storage
  , saveParam
  , loadParam

  -- * Data set
  , DataSet
  , mkGraph
  , mkSent

  -- * Error
  , netError

  -- * Training
  , trainProg
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat, natVal)
-- import           GHC.Natural (naturalToInt)
import qualified GHC.TypeNats as Nats

import           Control.Monad (guard, forM_)

import           Lens.Micro.TH (makeLenses)
import           Lens.Micro ((^.))

import           Data.Proxy (Proxy(..))
import           Data.Ord (comparing)
import qualified Data.List as L
import qualified Data.Text as T
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Array as A
import           Data.Binary (Binary)
import qualified Data.Binary as Bin
import qualified Data.ByteString.Lazy as B
import           Codec.Compression.Zlib (compress, decompress)

import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>), dot)
import qualified Numeric.LinearAlgebra.Static as LA

import           Net.Basic
import qualified Net.List as NL
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
import qualified GradientDescent as GD
import qualified GradientDescent.Momentum as Mom

import qualified Embedding.Dict as D


----------------------------------------------
-- Network Parameters
----------------------------------------------


-- | Parameters of the graph-structured network
--   * d -- embedding dimention size
--   * c -- number of classes
data Param d c = Param
  { _incM :: L d d
    -- ^ Incoming edges matrix
  , _incB :: R d
    -- ^ Incoming edges bias
  , _outM :: L d d
    -- ^ Outgoing edges matrix
  , _outB :: R d
    -- ^ Outgoing edges bias
  , _updateW :: L d d
    -- ^ Update matrix W
  , _updateU :: L d d
    -- ^ Update matrix U
  , _resetW :: L d d
    -- ^ Reset matrix W
  , _resetU :: L d d
    -- ^ Reset matrix U
  , _finalW :: L d d
    -- ^ Final matrix W
  , _finalU :: L d d
    -- ^ Final matrix U
  , _probM :: L
      c -- Output: probabilities for different categories
      d -- Size of the hidden state
    -- ^ Output probability matrix
--   , _probN :: FFN
--       300   -- Size of the hidden state
--       3   -- Hidden layer size
--       2   -- Output: single value (`2` due to a bug)
--     -- ^ Output probability FFN
    -- WARNING: it might be not enough to use a matrix here, since it is not
    -- able to transform a 0 vector into something which is not 0!
  , _arcM :: L d (d Nats.+ d)
    -- ^ A matrix to calculate arc embeddings based on node embeddings
  , _arcB :: R d
    -- ^ Bias for `arcM`
  } deriving (Generic, Binary)
  -- TODO: automatically deriving `Show` does not work 

instance (KnownNat d, KnownNat c) => BP.Backprop (Param d c)
makeLenses ''Param

-- instance (KnownNat d, KnownNat c) => Binary (Param d c) where
--   put Param{..} = do
--     Bin.put _incM
--     Bin.put _incB
--     Bin.put _outM
--     Bin.put _outB
--     Bin.put _updateW
--     Bin.put _updateU
--     Bin.put _resetW
--     Bin.put _resetU
--     Bin.put _finalW
--     Bin.put _finalU
--     Bin.put _probM
--     Bin.put _arcM
--     Bin.put _arcB
--   get = Param
--     <$> Bin.get
--     <*> Bin.get
--     <*> Bin.get
--     <*> Bin.get
--     <*> Bin.get
--     <*> Bin.get
--     <*> Bin.get
--     <*> Bin.get
--     <*> Bin.get
--     <*> Bin.get
--     <*> Bin.get
--     <*> Bin.get
--     <*> Bin.get


instance (KnownNat d, KnownNat c) => Mom.ParamSet (Param d c) where
  zero = Param
    (mat dpar dpar)
    (vec dpar)
    (mat dpar dpar)
    (vec dpar)
    (mat dpar dpar)
    (mat dpar dpar)
    (mat dpar dpar)
    (mat dpar dpar)
    (mat dpar dpar)
    (mat dpar dpar)
    (mat cpar dpar)
    (mat dpar (dpar*2))
    (vec dpar)
      where
        mat n m = LA.matrix (take (m*n) [0,0..])
        vec n   = LA.vector (take n [0,0..])
        dpar = proxyVal (Proxy :: Proxy d)
        cpar = proxyVal (Proxy :: Proxy c)
        proxyVal = fromInteger . toInteger . natVal
  add x y = Param
    { _incM = _incM x + _incM y
    , _incB = _incB x + _incB y
    , _outM = _outM x + _outM y
    , _outB = _outB x + _outB y
    , _updateW = _updateW x + _updateW y
    , _updateU = _updateU x + _updateU y
    , _resetW = _resetW x + _resetW y
    , _resetU = _resetU x + _resetU y
    , _finalW = _finalW x + _finalW y
    , _finalU = _finalU x + _finalU y
    , _probM = _probM x + _probM y
    , _arcM = _arcM x + _arcM y
    , _arcB = _arcB x + _arcB y
    }
  scale coef x = Param
    { _incM = scaleL $ _incM x
    , _incB = scaleR $ _incB x
    , _outM = scaleL $ _outM x
    , _outB = scaleR $ _outB x
    , _updateW = scaleL $ _updateW x
    , _updateU = scaleL $ _updateU x
    , _resetW = scaleL $ _resetW x
    , _resetU = scaleL $ _resetU x
    , _finalW = scaleL $ _finalW x
    , _finalU = scaleL $ _finalU x
    , _probM = scaleL $ _probM x
    , _arcM = scaleL $ _arcM x
    , _arcB = scaleR $ _arcB x
    } where
        scaleL = LA.dmmap (*coef)
        scaleR = LA.dvmap (*coef)
  size = BP.evalBP size


-- | Size (euclidean norm) of the network parameters
size
  :: (KnownNat d, KnownNat c, Reifies s W)
  => BVar s (Param d c)
  -> BVar s Double
size net =
  sqrt $ sum
    [ LBP.norm_2M (net ^^. incM) ^ 2
    , LBP.norm_2V (net ^^. incB) ^ 2
    , LBP.norm_2M (net ^^. outM) ^ 2
    , LBP.norm_2V (net ^^. outB) ^ 2
    , LBP.norm_2M (net ^^. updateW) ^ 2
    , LBP.norm_2M (net ^^. updateU) ^ 2
    , LBP.norm_2M (net ^^. resetW) ^ 2
    , LBP.norm_2M (net ^^. resetU) ^ 2
    , LBP.norm_2M (net ^^. finalW) ^ 2
    , LBP.norm_2M (net ^^. finalU) ^ 2
    , LBP.norm_2M (net ^^. probM) ^ 2
    , LBP.norm_2M (net ^^. arcM) ^ 2
    , LBP.norm_2V (net ^^. arcB) ^ 2
    ]


-- | Create a new, random network.
new :: (KnownNat d, KnownNat c) => Int -> Int -> IO (Param d c)
new d c = Param
  <$> matrix d d
  <*> vector d
  <*> matrix d d
  <*> vector d
  <*> matrix d d
  <*> matrix d d
  <*> matrix d d
  <*> matrix d d
  <*> matrix d d
  <*> matrix d d
  <*> matrix c d
  <*> matrix d (d*2)
  <*> vector d


-- | Local graph type
data Graph a = Graph
  { graphStr :: G.Graph
    -- ^ The underlying directed graph
  , graphInv :: G.Graph
    -- ^ Inversed (transposed) `graphStr`
  , labelMap :: M.Map G.Vertex a
    -- ^ Label assigned to a given vertex
  } deriving (Show, Eq, Ord, Functor)


-- | A graph arc (edge)
type Arc = (G.Vertex, G.Vertex)


-- | Run the network over a DAG labeled with input word embeddings.
run
  :: (KnownNat d, KnownNat c, Reifies s W)
  => BVar s (Param d c)
    -- ^ Graph network / params
  -> Int
    -- ^ Recursion depth
  -> Graph (BVar s (R d))
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s (R c))
    -- ^ Output map with (i) final hidden states and (ii) output values
run net depth graph =
  go (labelMap graph) (mkArcMap $ labelMap graph) depth
  where
    mkArcMap nodeMap = M.fromList $ do
      (v, w) <- graphArcs graph
      let hv = nodeMap M.! v
          hw = nodeMap M.! w
          -- TODO: try adding bias
          -- TODO: consider using feed-forward network
          -- NOTE: with `logistic` training didn't go well, but maybe you can
          --   try it again later
          he = (net ^^. arcM) #> (hv # hw) + (net ^^. arcB)
      return ((v, w), he)
    go prevNodeMap prevArcMap k
      | k <= 0 = M.fromList $ do
          (e, h) <- M.toList prevArcMap
          let x = softmax $ (net ^^. probM) #> h
          -- let x = elem0 $ FFN.run (net ^^. probN) h
          return (e, x)
      | otherwise =
          let
            attMap = M.fromList $ do
              v <- graphNodes graph
              let inc = sum $ do
                    w <- incoming v graph
                    let hw = prevArcMap M.! (w, v)
                        -- TODO: perhaps bias should not be included for each
                        -- adjacent edge, just once?
                        x = (net ^^. incM) #> hw + (net ^^. incB)
                    return x
              let out = sum $ do
                    w <- outgoing v graph
                    let hw = prevArcMap M.! (v, w)
                        -- TODO: perhaps bias should not be included for each
                        -- adjacent edge, just once?
                        x = (net ^^. outM) #> hw + (net ^^. outB)
                    return x
              -- TODO: here something different then sum (maybe concatenation?)
              -- could be possibly used
              return (v, inc + out)
            updateMap = M.fromList $ do
              v <- graphNodes graph
              let att = attMap M.! v
                  upd = sigma
                      $ (net ^^. updateW) #> att
                      + (net ^^. updateU) #> (prevNodeMap M.! v)
              return (v, upd)
            resetMap = M.fromList $ do
              v <- graphNodes graph
              let att = attMap M.! v
                  res = sigma
                      $ (net ^^. resetW) #> att
                      + (net ^^. resetU) #> (prevNodeMap M.! v)
              return (v, res)
            newHiddenTilda = M.fromList $ do
              v <- graphNodes graph
              let att = attMap M.! v
                  res = resetMap M.! v
                  hidPrev = prevNodeMap M.! v
                  result = tanh
                    ( (net ^^. finalW) #> att +
                      (net ^^. finalU) #> (res * hidPrev)
                    )
              return (v, result)
            newHidden = M.fromList $ do
              v <- graphNodes graph
              let upd = updateMap M.! v
                  hidPrev = prevNodeMap M.! v
                  hidTilda = newHiddenTilda M.! v
                  result = ((1 - upd)*hidPrev) + (upd*hidTilda)
              return (v, result)
          in
            go newHidden (mkArcMap newHidden) (k-1)


-- | Evaluate the network over a graph labeled with input word embeddings.
-- User-friendly (and without back-propagation) version of `run`.
eval
  :: (KnownNat d, KnownNat c)
  => Param d c
    -- ^ Graph network / params
  -> Int
    -- ^ Recursion depth
  -> Graph (R d)
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc (R c)
eval net depth graph =
  BP.evalBP0
    ( BP.collectVar
    $ run
        (BP.constVar net)
        depth
        (fmap BP.constVar graph)
    )


-- | Run the network on the test graph and print the resulting label map.
runTest
  :: (KnownNat d, KnownNat c)
  => Param d c
  -> Int
  -> Graph (R d)
  -> IO ()
runTest net depth graph =
  forM_ (M.toList $ eval net depth graph) $ \(e, v) ->
    print (e, LA.unwrap v)


----------------------------------------------
-- Serialization
----------------------------------------------


-- | Save the parameters in the given file.
saveParam
  :: (KnownNat d, KnownNat c)
  => FilePath
  -> Param d c
  -> IO ()
saveParam path =
  B.writeFile path . compress . Bin.encode


-- | Load the parameters from the given file.
loadParam
  :: (KnownNat d, KnownNat c)
  => FilePath
  -> IO (Param d c)
loadParam path =
  Bin.decode . decompress <$> B.readFile path
  -- B.writeFile path . compress . Bin.encode


----------------------------------------------
-- DataSet
----------------------------------------------


-- | DataSet: a list of (graph, target value map) pairs.
type DataSet d c =
  [ ( Graph (R d)
    , M.Map Arc (R c)
    )
  ]


-- | Create graph from lists of labeled nodes and edges.
mkGraph
  :: (KnownNat d)
  => [(G.Vertex, R d)]
    -- ^ Nodes with input vectors
  -> [(G.Vertex, G.Vertex)]
    -- ^ Edges
  -> Graph (R d)
mkGraph nodes arcs =
  graph
  where
    vertices = [v | (v, _) <- nodes]
    gStr = G.buildG (minimum vertices, maximum vertices) arcs
    lbMap = M.fromList nodes
    graph = Graph
      { graphStr = gStr
      , graphInv = G.transposeG gStr
      , labelMap = lbMap }


-- | Create an input sentence consisting of the given embedding vectors.
mkSent
  :: (KnownNat d)
  => D.Dict d
    -- ^ Embedding dictionary
  -> [T.Text]
    -- ^ Input words
  -> Graph (R d)
mkSent d ws =
  mkGraph
    [(i, vec w) | (i, w) <- zip [0..] ws]
    [(i, i+1) | i <- is]
  where
    vec = (d M.!)
    is = [0 .. length ws - 2]


----------------------------------------------
-- Error
----------------------------------------------


-- -- | Squared error between the target and the actual output.
-- errorOne
--   :: (Ord a, KnownNat n, Reifies s W)
--   => M.Map a (BVar s (R n))
--     -- ^ Target values
--   -> M.Map a (BVar s (R n))
--     -- ^ Output values
--   -> BVar s Double
-- errorOne target output = PB.sum . BP.collectVar $ do
--   (key, tval) <- M.toList target
--   let oval = output M.! key
--       err = tval - oval
--   return $ err `dot` err


-- | Cross entropy between the true and the artificial distributions
crossEntropy
  :: (KnownNat n, Reifies s W)
  => BVar s (R n)
    -- ^ Target ,,true'' distribution
  -> BVar s (R n)
    -- ^ Output ,,artificial'' distribution
  -> BVar s Double
crossEntropy p q =
  negate (p `dot` LBP.vmap' log q)


-- | Cross entropy between the target and the output values
errorOne
  :: (Ord a, KnownNat n, Reifies s W)
  => M.Map a (BVar s (R n))
    -- ^ Target values
  -> M.Map a (BVar s (R n))
    -- ^ Output values
  -> BVar s Double
errorOne target output = PB.sum . BP.collectVar $ do
  (key, tval) <- M.toList target
  let oval = output M.! key
  return $ crossEntropy tval oval


-- | Error on a dataset.
errorMany
  :: (Ord a, KnownNat n, Reifies s W)
  => [M.Map a (BVar s (R n))] -- ^ Targets
  -> [M.Map a (BVar s (R n))] -- ^ Outputs
  -> BVar s Double
errorMany targets outputs =
  go targets outputs
  where
    go ts os =
      case (ts, os) of
        (t:tr, o:or) -> errorOne t o + go tr or
        ([], []) -> 0
        _ -> error "errorMany: lists of different size"


-- | Network error on a given dataset.
netError
  :: (Reifies s W, KnownNat d, KnownNat c)
  => DataSet d c
  -> Int -- ^ Recursion depth
  -> BVar s (Param d c)
  -> BVar s Double
netError dataSet depth net =
  let
    inputs = map fst dataSet
    outputs = map (run net depth . fmap BP.auto) inputs
    targets = map (fmap BP.auto . snd) dataSet
  in  
    errorMany targets outputs -- + (size net * 0.01)


----------------------------------------------
-- Training
----------------------------------------------


-- -- | Train with a custom dataset.
-- trainWith gdCfg net =
--   Mom.gradDesc net gdCfg


-- | Progressive training
trainProg 
  :: (KnownNat d, KnownNat c)
  => (Int -> Mom.Config (Param d c))
    -- ^ Gradient descent config, depending on the chosen depth
  -> Int
    -- ^ Maximum depth
  -> Param d c
    -- ^ Initial params
  -> IO (Param d c)
trainProg gdCfg maxDepth =
  go 0
  where
    go depth net
      | depth > maxDepth =
          return net
      | otherwise = do
          putStrLn $ "# depth = " ++ show depth
          net' <- Mom.gradDesc net (gdCfg depth)
          go (depth+1) net'


----------------------------------------------
-- Graph Utils
----------------------------------------------


-- | Return the list of vertives in the graph.
graphNodes :: Graph a -> [G.Vertex]
graphNodes = G.vertices . graphStr


-- | Return the list of vertives in the graph.
graphArcs :: Graph a -> [Arc]
graphArcs = G.edges . graphStr


-- | Return the list of outgoing vertices.
outgoing :: G.Vertex -> Graph a -> [G.Vertex]
outgoing v Graph{..} =
  graphStr A.! v


-- | Return the list of incoming vertices.
incoming :: G.Vertex -> Graph a -> [G.Vertex]
incoming v Graph{..} =
  graphInv A.! v
