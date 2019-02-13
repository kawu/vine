{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE OverloadedStrings #-}


module Net.POS where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
import qualified GHC.TypeNats as Nats

import           Control.Monad (guard, forM_)

import           Lens.Micro.TH (makeLenses)
import           Lens.Micro ((^.))

import           Data.Ord (comparing)
import qualified Data.List as L
import qualified Data.Text as T
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Array as A

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

-- import Debug.Trace (trace)


----------------------------------------------
-- Recurrent DAG Network
----------------------------------------------


-- | Graph-structured network
--   * d -- dimention size
--   * c -- number of classes
data GraphNet d c = GraphNet
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
  } deriving (Show, Generic)

instance (KnownNat d, KnownNat c) => BP.Backprop (GraphNet d c)
makeLenses ''GraphNet

instance (KnownNat d, KnownNat c) => Mom.ParamSet (GraphNet d c) where
  zero = GraphNet
    (mat 300 300)
    (vec 300)
    (mat 300 300)
    (vec 300)
    (mat 300 300)
    (mat 300 300)
    (mat 300 300)
    (mat 300 300)
    (mat 300 300)
    (mat 300 300)
    (mat 5 300)
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
new :: (KnownNat d, KnownNat c) => Int -> Int -> IO (GraphNet d c)
new d c = GraphNet
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
  :: (KnownNat d, KnownNat c, Reifies s W)
  => BVar s (GraphNet d c)
    -- ^ Graph network / params
  -> Int
    -- ^ Recursion depth
  -> Graph (BVar s (R d))
    -- ^ Input graph labeled with initial hidden states
  -> M.Map G.Vertex (BVar s (R c))
    -- ^ Output map with (i) final hidden states and (ii) output values
run net depth graph =
  go (labelMap graph) depth
  where
    go prevHiddenMap k
      | k <= 0 = M.fromList $ do
          (v, h) <- M.toList prevHiddenMap
          -- TODO: logistic could be possibly replaced by something else!
          let x = softmax $ (net ^^. probM) #> h
          -- let x = elem0 $ FFN.run (net ^^. probN) h
          return (v, x)
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
  :: (KnownNat d, KnownNat c)
  => GraphNet d c
    -- ^ Graph network / params
  -> Int
    -- ^ Recursion depth
  -> Graph (R d)
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map G.Vertex (R c)
eval net depth graph =
  BP.evalBP0
    ( BP.collectVar
    $ run
        (BP.constVar net)
        depth
        (fmap BP.constVar graph)
    )


----------------------------------------------
-- Error
----------------------------------------------


-- | Dataset: a list of (graph, target value map) pairs.
type Dataset d c =
  [ ( Graph (R d)
    , M.Map G.Vertex (R c)
    )
  ]


-- | Squared error between the target and the actual output.
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
      err = tval - oval
  return $ err `dot` err


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
  => Dataset d c
  -> Int -- ^ Recursion depth
  -> BVar s (GraphNet d c)
  -> BVar s Double
netError dataSet depth net =
  let
    inputs = map fst dataSet
    outputs = map (run net depth . fmap BP.auto) inputs
    targets = map (fmap BP.auto . snd) dataSet
  in  
    errorMany targets outputs -- + (size net * 0.01)


----------------------------------------------
-- Training dataset
----------------------------------------------


-- | POS tags
noun, verb, adj, adv, det :: R 5
noun = LA.vector [1, 0, 0, 0, 0]
verb = LA.vector [0, 1, 0, 0, 0]
adj  = LA.vector [0, 0, 1, 0, 0]
adv  = LA.vector [0, 0, 0, 1, 0]
det  = LA.vector [0, 0, 0, 0, 1]


-- -- | Vocabulary, including a special UNK symbol (not useful for the moment?)
-- zero, one :: R 5
-- zero = LA.vector [0, 0, 0, 0, 1] -- NOTE: this `1` is important!
-- one  = LA.vector [1, 0, 0, 0, 0]


-- | Training dataset
trainData :: FilePath -> IO (Dataset 300 5)
trainData path = do
  -- Load the embedding dictionary
  d <- D.load False path
  let vec = (d M.!)
      john = vec "John"
      sleeps = vec "sleeps"
      eats = vec "eats"
      eat = vec "eat"
      well = vec "well"
      a = vec "a"
      an = vec "an"
      the = vec "the"
      pizza = vec "pizza"
      pizzas = vec "pizzas"
      cat = vec "cat"
      cats = vec "cats"
      tasty = vec "tasty"
      nice = vec "nice"
      break = vec "break"
      have = vec "have"
      has = vec "has"
  return $
    [ mkElem 
        [ (0, john) =>> noun 
        , (1, sleeps) =>> verb
        ]
        [(0, 1)]
    , mkElem 
        [ (0, john) =>> noun
        , (1, sleeps) =>> verb
        , (2, well) =>> adv
        ]
        [(0, 1), (1, 2)]
    , mkElem 
        [ (0, a) =>> det
        , (1, cat) =>> noun
        , (2, sleeps) =>> verb
        ]
        [(0, 1), (1, 2)]
    , mkElem 
        [ (0, a) =>> det
        , (1, cat) =>> noun
        , (2, eats) =>> verb
        , (3, a) =>> det
        , (4, pizza) =>> noun
        ]
        [(0, 1), (1, 2), (2, 3), (3, 4)]
    , mkElem
        [ (0, john) =>> noun
        , (1, eats) =>> verb
        , (2, a) =>> det
        , (3, tasty) =>> adj
        , (4, pizza) =>> noun
        ]
        [(0, 1), (1, 2), (2, 3), (3, 4)]
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
  :: (KnownNat d)
  => [(G.Vertex, R d)]
    -- ^ Nodes with input labels
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


-- | Create the input sentence consisting of embedding vectors.
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


-- | Run the network on the test graph and print the resulting label map.
runTest net depth graph =
  forM_ (M.toList $ eval net depth graph) $ \(x, v) ->
    print (x, LA.unwrap v)
  -- mapM_ print (M.toList $ eval net depth graph)


----------------------------------------------
-- Training
----------------------------------------------


-- | Train with a custom dataset.
trainWith dataSet depth net =
  Mom.gradDesc net (gdCfg dataSet depth)


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
train embPath depth net0 = do
  dataSet <- trainData embPath
  trainProg dataSet depth net0
  -- trainWith trainData 3


-----------------------------------------------------
-- Gradient descent
-----------------------------------------------------


-- | Gradient descent configuration
gdCfg dataSet depth = Mom.Config
  { iterNum = 100
  , gradient = BP.gradBP (netError dataSet depth)
  , quality = BP.evalBP (netError dataSet depth)
  , reportEvery = 10
  , gain0 = 0.05
  , tau = 50
  , gamma = 0.9
  }


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
