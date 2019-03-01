{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
-- {-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
-- {-# LANGUAGE OverloadedStrings #-}
-- {-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TupleSections #-}

-- To derive Binary for `Param`
{-# LANGUAGE DeriveAnyClass #-}

-- To make GHC automatically infer that `KnownNat d => KnownNat (d + d)`
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}


module Net.ArcGraph
  ( 
  -- * Network
    Config(..)
  , Param(..)
  -- , size
  , new
  , Graph(..)
  , Node(..)
  , Arc
  , run
  , eval
  -- , runTest

  -- * Storage
  , saveParam
  , loadParam

  -- * Data set
  , DataSet
  , Elem(..)
  -- , mkGraph
  -- , mkSent

  -- * Error
  , netError

  -- * Training
  -- , trainProg
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat) -- , natVal)
-- import           GHC.Natural (naturalToInt)
import qualified GHC.TypeNats as Nats

import           Control.Monad (forM_, forM)

import           Lens.Micro.TH (makeLenses)
-- import           Lens.Micro ((^.))

-- import           Data.Proxy (Proxy(..))
-- import           Data.Ord (comparing)
-- import qualified Data.List as L
import           Data.Maybe (catMaybes)
import qualified Data.Text as T
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Array as A
import           Data.Binary (Binary)
import qualified Data.Binary as Bin
import qualified Data.ByteString.Lazy as B
import           Codec.Compression.Zlib (compress, decompress)

-- import qualified Data.Map.Lens as ML
import           Control.Lens.At (ixAt)

import           Dhall (Interpret)
import qualified Data.Aeson as JSON

import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>), dot)
import qualified Numeric.LinearAlgebra.Static as LA

import           Net.Basic
-- import qualified Net.List as NL
-- import qualified Net.FeedForward as FFN
-- import           Net.FeedForward (FFN(..))
-- import qualified GradientDescent as GD
-- import qualified GradientDescent.Momentum as Mom
-- import qualified GradientDescent.Nestorov as Mom
-- import qualified GradientDescent.AdaDelta as Ada
import           Numeric.SGD.ParamSet (ParamSet)
import qualified Numeric.SGD.ParamSet as SGD

-- import qualified Embedding.Dict as D

import           Debug.Trace (trace)


----------------------------------------------
-- Network Parameters
----------------------------------------------


-- | Network configuration
data Config = Config
  { useWordAff :: Bool
    -- ^ Use word affinities
  , useArcLabels :: Bool
    -- ^ Use arc labels (UD dep rels)
  , useNodeLabels :: Bool
    -- ^ Use node labels (POS tags)
  , useBiaff :: Bool
    -- ^ Use ,,biaffine bias''
  , useUnordBiaff :: Bool
    -- ^ Use ,,unordered'' biaffinity.  The ,,standard'' biaffinity captures
    -- the bias of an ordered pair of words (represented by their embeddings),
    -- with the order determined by the direction of the dependency relation
    -- between them.  The ,,unordered'' version is not really unordered, it's
    -- rather that the order between two words is determined lexicographically
    -- (based on `nodeLex`).  This allows to capture MWEs occurrences with a
    -- reversed dependency direction, e.g., in the passive voice.
  } deriving (Generic)

instance JSON.FromJSON Config
instance JSON.ToJSON Config where
  toEncoding = JSON.genericToEncoding JSON.defaultOptions
instance Interpret Config


-- | Parameters of the graph-structured network
--   * d -- embedding dimention size
--   * c -- number of classes
--   * alb -- arc label
--   * nlb -- node label
data Param d c alb nlb = Param
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

  -- Only the fields that follow are actually important in case when depth = 0.

  , _probM :: L
      c -- Output: probabilities for different categories
      d -- Size of the hidden state
    -- ^ Output probability matrix
    -- WARNING: it might be not enough to use a matrix here, since it is not
    -- able to transform a 0 vector into something which is not 0!

--   , _potN :: FFN
--       (d Nats.+ d)
--       d -- Hidden layer size
--       d -- Output
--     -- ^ Potential FFN
--   , _potR :: R d
--     -- ^ Potential ,,feature vector''

  , _arcBiaff :: Maybe (L d (d Nats.+ d))
    -- ^ Biaffine bias
  , _arcUnordBiaff :: Maybe (L d (d Nats.+ d))
    -- ^ Unordered biaffine bias

  , _arcB :: R d
    -- ^ Default bias

  -- NOTE: We replace the single `_wordAff` with two word-related affinities.
  -- This is because the affinity of a word being a VMWE head will typically be
  -- different from the affinity of this word being a MWE dependent.
  --
  -- , _wordAff :: Maybe (L d d) -- `Nothing` means that not used
  -- -- ^ (Single) word affinity (NEW 27.02.2019)
  , _sourceAff :: Maybe (L d d) -- `Nothing` means that not used
    -- ^ Source word affinity (NEW 01.03.2019)
  , _targetAff :: Maybe (L d d) -- `Nothing` means that not used
    -- ^ Target word affinity (NEW 01.03.2019)

  , _arcLabelB :: Maybe (M.Map alb (R d))
    -- ^ Arc label (UD dependency relation) induced bias (NEW 28.02.2019)

  , _nodeLabelB :: Maybe (M.Map (nlb, nlb) (R d))
    -- ^ Arc label (UD dependency relation) induced bias (NEW 28.02.2019)
  } deriving (Generic, Binary)
  -- NOTE: automatically deriving `Show` does not work 

instance (KnownNat d, KnownNat c, Ord alb, Ord nlb)
  => BP.Backprop (Param d c alb nlb)
makeLenses ''Param

instance (KnownNat d, KnownNat c, Ord alb, Ord nlb)
  => ParamSet (Param d c alb nlb)


-- | Create a new, random network.
new
  :: (KnownNat d, KnownNat c, Ord alb, Ord nlb)
  => Int
  -> Int
  -> S.Set nlb
    -- ^ Set of node labels
  -> S.Set alb
    -- ^ Set of arc labels
  -> Config
  -> IO (Param d c alb nlb)
new d c nodeLabelSet arcLabelSet Config{..} = Param
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
  <*> matrix c d -- FFN.new (d*2) d d 
  -- <*> vector d
  <*> ( if useBiaff
           then Just <$> matrix d (d*2)
           else pure Nothing
      )
  <*> ( if useUnordBiaff
           then Just <$> matrix d (d*2)
           else pure Nothing
      )
  <*> vector d
  <*> ( if useWordAff
           then Just <$> matrix d d
           else pure Nothing
      )
  <*> ( if useWordAff
           then Just <$> matrix d d
           else pure Nothing
      )
  <*> ( if useArcLabels
           then Just <$> mkMap arcLabelSet (vector d)
           else pure Nothing
      )
  <*> ( if useNodeLabels
           then Just <$> mkMap (cart nodeLabelSet) (vector d)
           else pure Nothing
      )
  where
    mkMap keySet mkVal = fmap M.fromList .
      forM (S.toList keySet) $ \key -> do
        (key,) <$> mkVal
    -- cartesian product
    cart s = S.fromList $ do
      x <- S.toList s
      y <- S.toList s
      return (x, y)


-- | Local graph type
data Graph a b = Graph
  { graphStr :: G.Graph
    -- ^ The underlying directed graph
  , graphInv :: G.Graph
    -- ^ Inversed (transposed) `graphStr`
  , nodeLabelMap :: M.Map G.Vertex a
    -- ^ Label assigned to a given vertex
  , arcLabelMap :: M.Map Arc b
    -- ^ Label assigned to a given arc
  } deriving (Show, Eq, Ord, Generic, Binary)


-- | Node label mapping
nmap :: (a -> c) -> Graph a b -> Graph c b
nmap f g =
  g {nodeLabelMap = fmap f (nodeLabelMap g)}


-- | Arc label mapping
amap :: (b -> c) -> Graph a b -> Graph a c
amap f g =
  g {arcLabelMap = fmap f (arcLabelMap g)}


-- | A graph arc (edge)
type Arc = (G.Vertex, G.Vertex)


-- | Structured node
data Node dim nlb = Node
  { nodeEmb :: R dim
    -- ^ Node embedding vector
  , nodeLab :: nlb
    -- ^ Node label (e.g., POS tag)
  , nodeLex :: T.Text
    -- ^ Lexical content (used for ,,unordered'' biaffinity)
  } deriving (Show, Binary, Generic)


-- | Run the network over a graph labeled with input word embeddings.
run
  :: ( KnownNat d, KnownNat c
    , Ord alb, Show alb
    , Ord nlb, Show nlb
    , Reifies s W 
     )
  => BVar s (Param d c alb nlb)
    -- ^ Graph network / params
  -> Int
    -- ^ Recursion depth
  -- -> Graph (BVar s (R d)) b
  -- -> Graph (R d) b
  -> Graph (Node d nlb) alb
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s (R c))
--   -> M.Map Arc (BVar s Double)
    -- ^ Output map with output potential values
run net depth graph =
  let nodeMap =
        fmap (BP.constVar . nodeEmb) (nodeLabelMap graph)
   in go nodeMap (mkArcMap nodeMap) depth
  where
    mkArcMap nodeMap = M.fromList $ do
      (v, w) <- graphArcs graph
      let hv = nodeMap M.! v
          hw = nodeMap M.! w
          -- NOTE: consider using feed-forward network?
          -- NOTE: with `logistic` training didn't go well, but maybe you can
          --   try it again later
          biAff = do
            biaff <- BP.sequenceVar (net ^^. arcBiaff)
            return $ biaff #> (hv # hw)
          -- siAff = do
          --   aff <- BP.sequenceVar (net ^^. wordAff)
          --   return $ (aff #> hv) + (aff #> hw)
          unordBiAff = do
            biaff <- BP.sequenceVar (net ^^. arcUnordBiaff)
            vLex <- nodeLex <$> M.lookup v (nodeLabelMap graph)
            wLex <- nodeLex <$> M.lookup w (nodeLabelMap graph)
            return $ if vLex <= wLex
               then biaff #> (hv # hw)
               else biaff #> (hw # hv)
          srcAff = do
            aff <- BP.sequenceVar (net ^^. sourceAff)
            return $ aff #> hv
          trgAff = do
            aff <- BP.sequenceVar (net ^^. targetAff)
            return $ aff #> hw
          arcBias = do
            arcBiasMap <- BP.sequenceVar (net ^^. arcLabelB)
            let err = trace
                  ( "ArcGraph.run: unknown arc label ("
                  ++ show (M.lookup (v, w) (arcLabelMap graph))
                  ++ ")" ) 0
            return . maybe err id $ do
              arcLabel <- M.lookup (v, w) (arcLabelMap graph)
              arcBiasMap ^^? ixAt arcLabel
          nodePairBias = do
            nodeBiasMap <- BP.sequenceVar (net ^^. nodeLabelB)
            -- Construct the node labels here for the sake of error reporting
            let nodeLabels = do
                  vLabel <- nodeLab <$> M.lookup v (nodeLabelMap graph)
                  wLabel <- nodeLab <$> M.lookup v (nodeLabelMap graph)
                  return (vLabel, wLabel)
                err = trace
                  ( "ArcGraph.run: undefined node label(s) ("
                  ++ show nodeLabels
                  ++ ")" ) 0
            return . maybe err id $ do
              vLabel <- nodeLab <$> M.lookup v (nodeLabelMap graph)
              wLabel <- nodeLab <$> M.lookup v (nodeLabelMap graph)
              nodeBiasMap ^^? ixAt (vLabel, wLabel)
          bias = Just $ net ^^. arcB
          he = sum . catMaybes $
            [ biAff, unordBiAff, srcAff, trgAff
            , arcBias, nodePairBias, bias
            ]
      return ((v, w), he)
    go prevNodeMap prevArcMap k
--       | k <= 0 = M.fromList $ do
--           (p, q) <- graphArcs graph
--           let v = prevNodeMap M.! p
--               w = prevNodeMap M.! q
--           let x = (net ^^. potR) `dot`
--                   (FFN.run (net ^^. potN) (v # w))
--           return ((p, q), logistic x)
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
  :: ( KnownNat d, KnownNat c
     , Ord alb, Show alb
     , Ord nlb, Show nlb )
  => Param d c alb nlb
    -- ^ Graph network / params
  -> Int
    -- ^ Recursion depth
  -- -> Graph (R d) b
  -> Graph (Node d nlb) alb
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc (R c)
--   -> M.Map Arc Double
eval net depth graph =
  BP.evalBP0
    ( BP.collectVar
    $ run
        (BP.constVar net)
        depth
        graph -- (nmap BP.constVar graph)
    )


-- -- | Run the network on the test graph and print the resulting label map.
-- runTest
--   :: (KnownNat d, KnownNat c, Ord b)
--   => Param d c b
--   -> Int
--   -- -> Graph (R d) b
--   -> Graph (Node d ()) b
--   -> IO ()
-- runTest net depth graph =
--   forM_ (M.toList $ eval net depth graph) $ \(e, v) ->
--     print (e, LA.unwrap v)
-- --     print (e, v)


----------------------------------------------
-- Serialization
----------------------------------------------


-- | Save the parameters in the given file.
saveParam
  :: (KnownNat d, KnownNat c, Binary alb, Binary nlb)
  => FilePath
  -> Param d c alb nlb
  -> IO ()
saveParam path =
  B.writeFile path . compress . Bin.encode


-- | Load the parameters from the given file.
loadParam
  :: (KnownNat d, KnownNat c, Binary alb, Binary nlb)
  => FilePath
  -> IO (Param d c alb nlb)
loadParam path =
  Bin.decode . decompress <$> B.readFile path
  -- B.writeFile path . compress . Bin.encode


----------------------------------------------
-- DataSet
----------------------------------------------


-- | Dataset element
data Elem dim c arc nlb = Elem
  { graph :: Graph (Node dim nlb) arc
    -- ^ Input graph
  , labMap :: M.Map Arc (R c)
--   , labMap :: M.Map Arc Double
    -- ^ Target labels
  } deriving (Show, Generic, Binary)


-- | DataSet: a list of (graph, target value map) pairs.
type DataSet d c alb nlb = [Elem d c alb nlb]


-- -- | Create graph from lists of labeled nodes and edges.
-- mkGraph
--   :: (KnownNat d, Ord b)
--   => [(G.Vertex, R d)]
--     -- ^ Nodes with input vectors
--   -> [(Arc, b)]
--     -- ^ Edges with labels
--   -> Graph (R d) b
-- mkGraph nodes arcs =
--   graph
--   where
--     vertices = [v | (v, _) <- nodes]
--     gStr = G.buildG
--       (minimum vertices, maximum vertices) 
--       (map fst arcs)
--     lbMap = M.fromList nodes
--     graph = Graph
--       { graphStr = gStr
--       , graphInv = G.transposeG gStr
--       , nodeLabelMap = lbMap
--       , arcLabelMap = M.fromList arcs
--       }


-- -- | Create an input sentence consisting of the given embedding vectors.
-- mkSent
--   :: (KnownNat d)
--   => D.Dict d
--     -- ^ Embedding dictionary
--   -> [T.Text]
--     -- ^ Input words
--   -> Graph (R d) ()
-- mkSent d ws =
--   mkGraph
--     [(i, vec w) | (i, w) <- zip [0..] ws]
--     [(i, i+1) | i <- is]
--   where
--     vec = (d M.!)
--     is = [0 .. length ws - 2]


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


-- -- | Cross entropy between the true and the artificial distributions
-- crossEntropy
--   :: (Reifies s W)
--   => BVar s Double
--     -- ^ Target ,,true'' MWE probability
--   -> BVar s Double
--     -- ^ Output ,,artificial'' MWE probability
--   -> BVar s Double
-- crossEntropy p q
--   | p < 0 || p > 1 || q < 0 || q > 1 =
--       error "crossEntropy doesn't make sense"
--   | otherwise = negate
--       ( p0 * log q0
--       + p1 * log q1
--       )
--   where
--     p1 = p
--     p0 = 1 - p1
--     q1 = q
--     q0 = 1 - q1


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
--   => M.Map a (BVar s Double)
    -- ^ Target values
  -> M.Map a (BVar s (R n))
--   -> M.Map a (BVar s Double)
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
--   => [M.Map a (BVar s Double)] -- ^ Targets
--   -> [M.Map a (BVar s Double)] -- ^ Outputs
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
  :: ( Reifies s W, KnownNat d, KnownNat c
     , Ord alb, Show alb, Ord nlb, Show nlb
     )
  => DataSet d c alb nlb
  -> Int -- ^ Recursion depth
  -> BVar s (Param d c alb nlb)
  -> BVar s Double
netError dataSet depth net =
  let
    inputs = map graph dataSet
    -- outputs = map (run net depth . nmap BP.auto) inputs
    outputs = map (run net depth) inputs
    targets = map (fmap BP.auto . labMap) dataSet
  in  
    errorMany targets outputs -- + (size net * 0.01)


----------------------------------------------
-- Training
----------------------------------------------


-- -- | Train with a custom dataset.
-- trainWith gdCfg net =
--   Mom.gradDesc net gdCfg


-- -- | Progressive training
-- trainProg 
--   :: (KnownNat d)
--   => (Int -> Mom.Config (Param d))
--     -- ^ Gradient descent config, depending on the chosen depth
--   -> Int
--     -- ^ Maximum depth
--   -> Param d
--     -- ^ Initial params
--   -> IO (Param d)
-- trainProg gdCfg maxDepth =
--   go 0
--   where
--     go depth net
--       | depth > maxDepth =
--           return net
--       | otherwise = do
--           putStrLn $ "# depth = " ++ show depth
--           net' <- Mom.gradDesc net (gdCfg depth)
--           go (depth+1) net'


----------------------------------------------
-- Graph Utils
----------------------------------------------


-- | Return the list of vertives in the graph.
graphNodes :: Graph a b -> [G.Vertex]
graphNodes = G.vertices . graphStr


-- | Return the list of vertives in the graph.
graphArcs :: Graph a b -> [Arc]
graphArcs = G.edges . graphStr


-- | Return the list of outgoing vertices.
outgoing :: G.Vertex -> Graph a b -> [G.Vertex]
outgoing v Graph{..} =
  graphStr A.! v


-- | Return the list of incoming vertices.
incoming :: G.Vertex -> Graph a b -> [G.Vertex]
incoming v Graph{..} =
  graphInv A.! v


----------------------------------------------
-- Maybe param
----------------------------------------------


-- instance (ParamSet a) => ParamSet (Maybe a) where
--   zero = fmap SGD.zero
--   pmap f = fmap (SGD.pmap f)
--   add x y
--     | keySetEq x y = M.unionWith SGD.add x y
--     | otherwise = error "ArcGraph: different key set"
--   sub x y
--     | keySetEq x y = M.unionWith SGD.sub x y
--     | otherwise = error "ArcGraph: different key set"
--   mul x y
--     | keySetEq x y = M.unionWith SGD.mul x y
--     | otherwise = error "ArcGraph: different key set"
--   div x y
--     | keySetEq x y = M.unionWith SGD.div x y
--     | otherwise = error "ArcGraph: different key set"
--   norm_2 = sqrt . sum . map ((^2) . SGD.norm_2)  . M.elems


----------------------------------------------
-- Parameter Map
----------------------------------------------


-- instance (Ord k, ParamSet a) => ParamSet (M.Map k a) where
--   zero = fmap SGD.zero
--   pmap f = fmap (SGD.pmap f)
--   add x y
--     | keySetEq x y = M.unionWith SGD.add x y
--     | otherwise = error "ArcGraph: different key set"
--   sub x y
--     | keySetEq x y = M.unionWith SGD.sub x y
--     | otherwise = error "ArcGraph: different key set"
--   mul x y
--     | keySetEq x y = M.unionWith SGD.mul x y
--     | otherwise = error "ArcGraph: different key set"
--   div x y
--     | keySetEq x y = M.unionWith SGD.div x y
--     | otherwise = error "ArcGraph: different key set"
--   norm_2 = sqrt . sum . map ((^2) . SGD.norm_2)  . M.elems
-- 
-- 
-- -- | The two maps have equal sets?
-- keySetEq :: (Eq k) => M.Map k v -> M.Map k v' -> Bool
-- keySetEq x y =
--   M.keys x == M.keys y
