{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}


module Net.Graph where


import           Control.Monad (guard)

import           GHC.Generics (Generic)

import           Lens.Micro.TH (makeLenses)
import           Lens.Micro ((^.))

import           Data.Ord (comparing)
import qualified Data.List as L
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Array as A

import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>))
import qualified Numeric.LinearAlgebra.Static as LA

import           Net.Basic
import qualified Net.List as NL
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
import qualified GradientDescent as GD
import qualified GradientDescent.Momentum as Mom

-- import Debug.Trace (trace)


----------------------------------------------
-- Recurrent DAG Network
----------------------------------------------


-- | Graph-structured network; Dimention size = 5
data GraphNet = GraphNet
  { _incM :: L 5 5
    -- ^ Incoming edges matrix
  , _incB :: R 5
    -- ^ Incoming edges bias
  , _outM :: L 5 5
    -- ^ Outgoing edges matrix
  , _outB :: R 5
    -- ^ Outgoing edges bias
  , _updateW :: L 5 5
    -- ^ Update matrix W
  , _updateU :: L 5 5
    -- ^ Update matrix U
  , _resetW :: L 5 5
    -- ^ Reset matrix W
  , _resetU :: L 5 5
    -- ^ Reset matrix U
  , _finalW :: L 5 5
    -- ^ Final matrix W
  , _finalU :: L 5 5
    -- ^ Final matrix U
  , _probM :: L
      2   -- Output: single value (`2` due to a bug)
      5   -- Size of the hidden state
    -- ^ Output probability matrix
--   , _probN :: FFN
--       5   -- Size of the hidden state
--       3   -- Hidden layer size
--       2   -- Output: single value (`2` due to a bug)
--     -- ^ Output probability FFN
    -- WARNING: it might be not enough to use a matrix here, since it is not
    -- able to transform a 0 vector into something which is not 0!
  } deriving (Show, Generic)

instance BP.Backprop GraphNet
makeLenses ''GraphNet

instance Mom.ParamSet GraphNet where
  zero = GraphNet
    (mat 5 5)
    (vec 5)
    (mat 5 5)
    (vec 5)
    (mat 5 5)
    (mat 5 5)
    (mat 5 5)
    (mat 5 5)
    (mat 5 5)
    (mat 5 5)
    (mat 2 5)
      where
        mat n m = LA.matrix (take (m*n) [0,0..])
        vec n   = LA.vector (take n [0,0..])
  add x y = GraphNet
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
    }
  scale coef x = GraphNet
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
    } where
        scaleL = LA.dmmap (*coef)
        scaleR = LA.dvmap (*coef)
  size = BP.evalBP size


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
    ]


-- | Create a new, random network
new :: IO GraphNet
new = GraphNet
  <$> matrix 5 5
  <*> vector 5
  <*> matrix 5 5
  <*> vector 5
  <*> matrix 5 5
  <*> matrix 5 5
  <*> matrix 5 5
  <*> matrix 5 5
  <*> matrix 5 5
  <*> matrix 5 5
  <*> matrix 2 5
  -- <*> FFN.new 5 3 2


-- | Local graph type
data Graph a = Graph
  { graphStr :: G.Graph
    -- ^ The underlying directed graph
  , graphInv :: G.Graph
    -- ^ Inversed (transposed) `graphStr`
  , labelMap :: M.Map G.Vertex a
    -- ^ Label assigned to a given vertex
  } deriving (Show, Eq, Ord, Functor)


-- | Run the network over a DAG labeled with input word embeddings.
run
  :: (Reifies s W)
  => BVar s GraphNet
    -- ^ Graph network / params
  -> Int
    -- ^ Recursion depth
  -> Graph (BVar s (R 5))
    -- ^ Input graph labeled with initial hidden states
  -> M.Map G.Vertex (BVar s (R 5), BVar s Double)
    -- ^ Output map with (i) final hidden states and (ii) output values
run net depth graph =
  go (labelMap graph) depth
  where
    go prevHiddenMap k
      | k <= 0 = M.fromList $ do
          (v, h) <- M.toList prevHiddenMap
          -- TODO: logistic could be possibly replaced by something else!
          let x = logistic . elem0 $ (net ^^. probM) #> h
          -- let x = elem0 $ FFN.run (net ^^. probN) h
          return (v, (h, x))
      | otherwise =
          let
            attMap = M.fromList $ do
              v <- graphNodes graph
              let inc = sum $ do
                    w <- incoming v graph
                    let hw = prevHiddenMap M.! w
                        x = (net ^^. incM) #> hw + (net ^^. incB)
                    return x
              let out = sum $ do
                    w <- outgoing v graph
                    let hw = prevHiddenMap M.! w
                        x = (net ^^. outM) #> hw + (net ^^. outB)
                    return x
              return (v, inc + out)
            updateMap = M.fromList $ do
              v <- graphNodes graph
              let att = attMap M.! v
                  upd = sigma
                      $ (net ^^. updateW) #> att
                      + (net ^^. updateU) #> (prevHiddenMap M.! v)
              return (v, upd)
            resetMap = M.fromList $ do
              v <- graphNodes graph
              let att = attMap M.! v
                  res = sigma
                      $ (net ^^. resetW) #> att
                      + (net ^^. resetU) #> (prevHiddenMap M.! v)
              return (v, res)
            newHiddenTilda = M.fromList $ do
              v <- graphNodes graph
              let att = attMap M.! v
                  res = resetMap M.! v
                  hidPrev = prevHiddenMap M.! v
                  result = tanh
                    ( (net ^^. finalW) #> att +
                      (net ^^. finalU) #> (res * hidPrev)
                    )
              return (v, result)
            newHidden = M.fromList $ do
              v <- graphNodes graph
              let upd = updateMap M.! v
                  hidPrev = prevHiddenMap M.! v
                  hidTilda = newHiddenTilda M.! v
                  result = ((1 - upd)*hidPrev) + (upd*hidTilda)
              return (v, result)
          in
            go newHidden (k-1)


-- | Evaluate the network over a graph labeled with input word embeddings.
-- User-friendly (and without back-propagation) version of `run`.
eval
  :: GraphNet
    -- ^ Graph network / params
  -> Int
    -- ^ Recursion depth
  -> Graph (R 5)
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map G.Vertex Double
eval net depth graph =
  BP.evalBP0
    ( BP.collectVar
    . fmap snd
    $ run
        (BP.constVar net)
        depth
        (fmap BP.constVar graph)
    )


----------------------------------------------
-- Error
----------------------------------------------


-- | Dataset: a list of (graph, target value map) pairs.
type Dataset =
  [ ( Graph (R 5)
    , M.Map G.Vertex Double )
  ]


-- | Squared error between the target and the actual output.
errorOne
  :: (Ord a, Reifies s W)
  => M.Map a (BVar s Double)
    -- ^ Target values
  -> M.Map a (BVar s Double)
    -- ^ Output values
  -> BVar s Double
errorOne target output = PB.sum . BP.collectVar $ do
  (key, tval) <- M.toList target
  let oval = output M.! key
  return $ (tval - oval) ^ 2


-- | Error on a dataset.
errorMany
  :: (Ord a, Reifies s W)
  => [M.Map a (BVar s Double)] -- ^ Targets
  -> [M.Map a (BVar s Double)] -- ^ Outputs
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
  :: (Reifies s W)
  => Dataset
  -> Int -- ^ Recursion depth
  -> BVar s GraphNet
  -> BVar s Double
netError dataSet depth net =
  let
    inputs = map fst dataSet
    outputs = map (fmap snd . run net depth . fmap BP.auto) inputs
    targets = map (fmap BP.auto . snd) dataSet
  in  
    errorMany targets outputs + (size net * 0.01)


----------------------------------------------
-- Training dataset
----------------------------------------------


-- -- | Vocabulary, including a special UNK symbol (not useful for the moment?)
-- noun, verb, adj, adv, unk :: R 5
-- noun = LA.vector [1, 0, 0, 0, 0]
-- verb = LA.vector [0, 1, 0, 0, 0]
-- adj  = LA.vector [0, 0, 1, 0, 0]
-- adv  = LA.vector [0, 0, 0, 1, 0]
-- unk  = LA.vector [0, 0, 0, 0, 1]


-- | Vocabulary, including a special UNK symbol (not useful for the moment?)
zero, one :: R 5
zero = LA.vector [0, 0, 0, 0, 1] -- NOTE: this `1` is important!
one  = LA.vector [1, 0, 0, 0, 0]


-- | Training dataset
trainData :: Dataset
trainData =
  [ mkElem 
      [(0, zero) =>> 0]
      []
  , mkElem [(0, one)  =>> 1] []
  , mkElem [(0, zero) =>> 0, (1, zero) =>> 0] [(0, 1)]
  , mkElem [(0, zero) =>> 0, (1, zero) =>> 0] [(1, 0)]
  , mkElem [(0, zero) =>> 0, (1, one)  =>> 1] [(0, 1)]
  , mkElem [(0, zero) =>> 1, (1, one)  =>> 1] [(1, 0)]
  , mkElem [(0, zero) =>> 0, (1, zero) =>> 0] [(0, 1), (1, 0)]
  , mkElem [(0, one)  =>> 1, (1, zero) =>> 1] [(0, 1), (1, 0)]
  , mkElem [(0, zero) =>> 1, (1, one)  =>> 1] [(0, 1), (1, 0)]
  , mkElem 
      [ (0, one)  =>> 1
      , (1, zero) =>> 1
      , (2, zero) =>> 1
      , (3, zero) =>> 1
      ] [(0, 1), (1, 2), (2, 3)]
  , mkElem 
      [ (0, zero) =>> 0
      , (1, zero) =>> 0
      , (2, one)  =>> 1
      , (3, zero) =>> 1
      ] [(0, 1), (1, 2), (2, 3)]
  , mkElem
      [ (0, zero)  =>> 0
      , (1, one)   =>> 1
      , (2, zero)  =>> 1
      , (3, zero)  =>> 1
      , (4, zero)  =>> 0
      , (5, zero)  =>> 0
      , (6, zero)  =>> 0
      , (7, zero)  =>> 1
      , (8, zero)  =>> 1
      , (9, zero)  =>> 0
      , (10, zero) =>> 1
      ]
      [ (0, 1), (1, 2), (2, 3), (3, 7)
      , (4, 0), (5, 1), (6, 3), (7, 8)
      , (9, 8), (3, 10), (10, 1)
      ]
  , mkElem
      [ (0, zero)  =>> 0
      , (1, zero)  =>> 0
      , (2, zero)  =>> 0
      ]
      [ (0, 1), (1, 2)
      ]
  , mkElem
      [ (0, one)   =>> 1
      , (1, zero)  =>> 1
      , (2, zero)  =>> 1
      ]
      [ (0, 1), (1, 2)
      ]
  , mkElem
      [ (0, zero)  =>> 0
      , (1, zero)  =>> 0
      , (2, zero)  =>> 0
      , (3, zero)  =>> 0
      ]
      [ (0, 1), (0, 2), (0, 3)
      , (1, 0), (1, 2), (1, 3)
      , (2, 0), (2, 1), (2, 3)
      , (3, 0), (3, 1), (3, 2)
      ]
  , mkElem
      [ (0, zero)  =>> 1
      , (1, zero)  =>> 1
      , (2, zero)  =>> 1
      , (3, one)   =>> 1
      ]
      [ (0, 1), (0, 2), (0, 3)
      , (1, 0), (1, 2), (1, 3)
      , (2, 0), (2, 1), (2, 3)
      , (3, 0), (3, 1), (3, 2)
      ]
  ]
  where
    (=>>) (v, h) x = (v, h, x)
    mkElem nodes arcs =
      (graph, valMap)
      where
        vertices = [v | (v, _, _) <- nodes]
        gStr = G.buildG (minimum vertices, maximum vertices) arcs
        lbMap = M.fromList [(v, h) | (v, h, _) <- nodes]
        graph = Graph
          { graphStr = gStr
          , graphInv = G.transposeG gStr
          , labelMap = lbMap }
        valMap = M.fromList [(v, x) | (v, _, x) <- nodes]


-- | Create graph from lists of labeled nodes and edges.
mkGraph
  :: [(G.Vertex, R 5)]
    -- ^ Nodes with input labels
  -> [(G.Vertex, G.Vertex)]
    -- ^ Edges
  -> Graph (R 5)
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


-- | Run the network on the test graph and print the resulting label map.
runTest net depth graph =
  mapM_ print (M.toList $ eval net depth graph)


----------------------------------------------
-- Training
----------------------------------------------


-- | Train with a custom dataset.
trainWith dataSet depth net =
  Mom.gradDesc net (momCfg dataSet depth)


-- | Progressive training
trainProg dataSet maxDepth =
  go 0
  where
    go depth net
      | depth > maxDepth = 
          return net
      | otherwise = do
          putStrLn $ "# depth = " ++ show depth
          net' <- trainWith dataSet depth net
          go (depth+1) net'


-- | Train with the default dataset.
train =
  trainProg trainData
  -- trainWith trainData 3


-----------------------------------------------------
-- Gradient descent
-----------------------------------------------------


-- | Gradient descent configuration
momCfg dataSet depth = Mom.Config
  { iterNum = 1000
  , gradient = BP.gradBP (netError dataSet depth)
  , quality = BP.evalBP (netError dataSet depth)
  , reportEvery = 100
  , gain0 = 0.1
  , tau = 100
  , gamma = 0.9
  }


-- -- | Gradient descent configuration
-- gdCfg dataSet depth = GD.Config
--   { iterNum = 1000
--   , scaleCoef = 0.01
--   , gradient = BP.gradBP (netError dataSet depth)
--   , substract = subNet
--   , quality = BP.evalBP (netError dataSet depth)
--   , reportEvery = 100
--   }


-- -- | Substract the second network (multiplied by the given coefficient) from
-- -- the first one.
-- subNet x y coef = GraphNet
--   { _incM = _incM x - scaleL (_incM y)
--   , _incB = _incB x - scaleR (_incB y)
--   , _outM = _outM x - scaleL (_outM y)
--   , _outB = _outB x - scaleR (_outB y)
--   , _updateW = _updateW x - scaleL (_updateW y)
--   , _updateU = _updateU x - scaleL (_updateU y)
--   , _resetW = _resetW x - scaleL (_resetW y)
--   , _resetU = _resetU x - scaleL (_resetU y)
--   , _finalW = _finalW x - scaleL (_finalW y)
--   , _finalU = _finalU x - scaleL (_finalU y)
--   , _probM = _probM x - scaleL (_probM y)
--   -- , _probN = FFN.substract (_probN x) (_probN y) coef
--   }
--   where
--     scaleL = LA.dmmap (*coef)
--     scaleR = LA.dvmap (*coef)


----------------------------------------------
-- Graph Utils
----------------------------------------------


-- | Return the list of vertives in the graph.
graphNodes :: Graph a -> [G.Vertex]
graphNodes = G.vertices . graphStr


-- | Return the list of outgoing vertices.
outgoing :: G.Vertex -> Graph a -> [G.Vertex]
outgoing v Graph{..} =
  graphStr A.! v


-- | Return the list of incoming vertices.
incoming :: G.Vertex -> Graph a -> [G.Vertex]
incoming v Graph{..} =
  graphInv A.! v
