{-# LANGUAGE CPP #-}
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

{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
-- -- JW: needed for 'instance BiComp ArcBias'
-- {-# LANGUAGE PolyKinds #-}

{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE Rank2Types #-}

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
import           Control.Lens (Lens)
import           Control.DeepSeq (NFData)

import           Dhall (Interpret)
import qualified Data.Aeson as JSON

import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, (^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>), dot)
import qualified Numeric.LinearAlgebra.Static as LA

import           Net.Basic
-- import qualified Net.List as NL
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
-- import qualified GradientDescent as GD
-- import qualified GradientDescent.Momentum as Mom
-- import qualified GradientDescent.Nestorov as Mom
-- import qualified GradientDescent.AdaDelta as Ada
import           Numeric.SGD.ParamSet (ParamSet)
import qualified Numeric.SGD.ParamSet as SGD

-- import qualified Embedding.Dict as D

import           Debug.Trace (trace)


----------------------------------------------
-- Graph
----------------------------------------------


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


----------------------------------------------
-- Config
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


----------------------------------------------
-- Type voodoo
----------------------------------------------


-- | Custom pair, with Backprop and ParamSet instances, and nice Backprop
-- pattern.
data a :& b = !a :& !b
  deriving (Show, Generic)
infixr 2 :&

instance (NFData a, NFData b) => NFData (a :& b)
instance (Backprop a, Backprop b) => Backprop (a :& b)
instance (ParamSet a, ParamSet b) => ParamSet (a :& b)

pattern (:&&) :: (Backprop a, Backprop b, Reifies z W)
              => BVar z a -> BVar z b -> BVar z (a :& b)
pattern x :&& y <- (\xy -> (xy ^^. t1, xy ^^. t2)->(x, y))
  where
    (:&&) = BP.isoVar2 (:&) (\case x :& y -> (x, y))
{-# COMPLETE (:&&) #-}


t1 :: Lens (a :& b) (a' :& b) a a'
t1 f (x :& y) = (:& y) <$> f x

t2 :: Lens (a :& b) (a :& b') b b'
t2 f (x :& y) = (x :&) <$> f y


-- type Model p a b = forall z. Reifies z W
--                 => BVar z p
--                 -> BVar z a
--                 -> BVar z b


----------------------------------------------
-- New
----------------------------------------------


class New a b p where
  new'
    :: Int
      -- ^ Embedding dimension (TODO: could be get rid of?)
    -> S.Set a
      -- ^ Set of node labels
    -> S.Set b
      -- ^ Set of arc labels
    -> IO p

instance (New a b p1, New a b p2) => New a b (p1 :& p2) where
  new' d xs ys = do
    p1 <- new' d xs ys
    p2 <- new' d xs ys
    return (p1 :& p2)


----------------------------------------------
-- Components
----------------------------------------------


-- | Biaffinity component
class Backprop comp => BiComp dim a b comp where
  runBiComp 
    :: (Reifies s W)
    => Graph (Node dim a) b
    -> Arc
    -> BVar s comp
    -> BVar s Double

instance (BiComp dim a b comp1, BiComp dim a b comp2)
  => BiComp dim a b (comp1 :& comp2) where
  runBiComp graph arc (comp1 :&& comp2) =
    runBiComp graph arc comp1 + runBiComp graph arc comp2


----------------------------------------------
----------------------------------------------


-- | Global bias
newtype Bias = Bias
  { _biasVal :: Double
  } deriving (Generic, Binary)

instance Backprop Bias
makeLenses ''Bias

instance New a b Bias where
  new' _ _ _ = pure (Bias 0)

instance BiComp dim a b Bias where
  runBiComp _ _ bias = bias ^^. biasVal


----------------------------------------------
----------------------------------------------


-- | Arc label bias
newtype ArcBias b = ArcBias
  { _arcBiasMap :: M.Map b Double
  } deriving (Generic, Binary)

instance (Ord b) => Backprop (ArcBias b)
makeLenses ''ArcBias

instance (Ord b) => New a b (ArcBias b) where
  new' _ _ arcLabelSet =
    -- TODO: could be simplified...
    ArcBias <$> mkMap arcLabelSet (pure 0)
    where
      mkMap keySet mkVal = fmap M.fromList .
        forM (S.toList keySet) $ \key -> do
          (key,) <$> mkVal

instance (Ord b, Show b) => BiComp dim a b (ArcBias b) where
  runBiComp graph (v, w) arcBias =
    let err = trace
          ( "ArcGraph.run: unknown arc label ("
          ++ show (M.lookup (v, w) (arcLabelMap graph))
          ++ ")" ) 0
     in maybe err id $ do
          arcLabel <- M.lookup (v, w) (arcLabelMap graph)
          arcBias ^^. arcBiasMap ^^? ixAt arcLabel


----------------------------------------------
----------------------------------------------


-- | Biaffinity component
data Biaff d h = Biaff
  { _biaffN :: FFN
      (d Nats.+ d)
      -- TODO: make hidden layer larger?  The example with the grenade library
      -- suggests this.
      h -- Hidden layer size
      d -- Output
    -- ^ Potential FFN
  , _biaffV :: R d
    -- ^ Potential ,,feature vector''
  } deriving (Generic, Binary)

instance (KnownNat dim, KnownNat h) => Backprop (Biaff dim h)
makeLenses ''Biaff

instance (KnownNat dim, KnownNat h) => BiComp dim a b (Biaff dim h) where
  runBiComp graph (v, w) bia =
     let nodeMap = fmap
           (BP.constVar . nodeEmb)
           (nodeLabelMap graph)
         hv = nodeMap M.! v
         hw = nodeMap M.! w
      in (bia ^^. biaffV) `dot`
           (FFN.run (bia ^^. biaffN) (hv # hw))


----------------------------------------------
-- Network Parameters
----------------------------------------------


-- | Parameters of the graph-structured network
--   * d -- embedding dimention size
--   * c -- number of classes
--   * alb -- arc label
--   * nlb -- node label
-- data Param d c alb nlb = Param
data Param d alb nlb = Param
  {
--     _probM :: L
--       c -- Output: probabilities for different categories
--       d -- Size of the hidden state
--     -- ^ Output probability matrix
--     -- WARNING: it might be not enough to use a matrix here, since it is not
--     -- able to transform a 0 vector into something which is not 0!

--     _probN :: FFN
--       d -- Hidden arc representation
--       d -- Hidden layer size
--       c -- Output
--     -- ^ Potential FFN

    _potN :: FFN
      (d Nats.+ d)
      -- TODO: make hidden layer larger?  The example with the grenade library
      -- suggests this.
      d -- Hidden layer size
      d -- Output
    -- ^ Potential FFN
  , _potR :: R d
    -- ^ Potential ,,feature vector''

--   , _arcBiaff :: Maybe (L d (d Nats.+ d))
--     -- ^ Biaffine bias
--   , _arcUnordBiaff :: Maybe (L d (d Nats.+ d))
--     -- ^ Unordered biaffine bias

--   , _arcB :: R d
--     -- ^ Default bias

  , _arcB :: Double
    -- ^ Default bias

--   -- NOTE: We replace the single `_wordAff` with two word-related affinities.
--   -- This is because the affinity of a word being a VMWE head will typically be
--   -- different from the affinity of this word being a MWE dependent.
--   --
--   -- , _wordAff :: Maybe (L d d) -- `Nothing` means that not used
--   -- -- ^ (Single) word affinity (NEW 27.02.2019)
--   , _sourceAff :: Maybe (L d d) -- `Nothing` means that not used
--     -- ^ Source word affinity (NEW 01.03.2019)
--   , _targetAff :: Maybe (L d d) -- `Nothing` means that not used
--     -- ^ Target word affinity (NEW 01.03.2019)

  , _arcLabelB :: Maybe (M.Map alb Double)
    -- ^ Arc label (UD dependency relation) induced bias (NEW 28.02.2019)

--   , _nodeLabelB :: Maybe (M.Map (nlb, nlb) Double)
--     -- ^ Arc label (UD dependency relation) induced bias (NEW 28.02.2019)

--   , _nodeLabelB :: Maybe (M.Map (nlb, nlb) (R d))
--     -- ^ Arc label (UD dependency relation) induced bias (NEW 28.02.2019)
  } deriving (Generic, Binary)
  -- NOTE: automatically deriving `Show` does not work 

instance (KnownNat d, Ord alb, Ord nlb)
  => Backprop (Param d alb nlb)
makeLenses ''Param

instance (KnownNat d, Ord alb, Ord nlb)
  => ParamSet (Param d alb nlb)

instance (KnownNat d, NFData alb, NFData nlb)
  => NFData (Param d alb nlb)


-- | Create a new, random network.
new
  :: (KnownNat d, Ord alb, Ord nlb)
  => Int
  -> S.Set nlb
    -- ^ Set of node labels
  -> S.Set alb
    -- ^ Set of arc labels
  -> Config
  -> IO (Param d alb nlb)
new d nodeLabelSet arcLabelSet Config{..} = Param
  -- <$> FFN.new d d c -- matrix c d
  <$> FFN.new (d*2) d d
  <*> vector d
  <*> pure 0
--   <*> ( if useBiaff
--            then Just <$> matrix d (d*2)
--            else pure Nothing
--       )
-- --   <*> matrix d (d*2)
--   <*> ( if useUnordBiaff
--            then Just <$> matrix d (d*2)
--            else pure Nothing
--       )
--   <*> vector d
--   <*> ( if useWordAff
--            then Just <$> matrix d d
--            else pure Nothing
--       )
--   <*> ( if useWordAff
--            then Just <$> matrix d d
--            else pure Nothing
--       )
--   <*> ( if useArcLabels
--            then Just <$> mkMap arcLabelSet (vector d)
--            else pure Nothing
--       )
  <*> ( if useArcLabels
           then Just <$> mkMap arcLabelSet (pure 0)
           else pure Nothing
      )
-- --   <*> ( if useNodeLabels
-- --            then Just <$> mkMap (cart nodeLabelSet) (vector d)
-- --            else pure Nothing
-- --       )
--   <*> ( if useNodeLabels
--            then Just <$> mkMap (cart nodeLabelSet) (pure 0)
--            else pure Nothing
--       )
  where
    mkMap keySet mkVal = fmap M.fromList .
      forM (S.toList keySet) $ \key -> do
        (key,) <$> mkVal
    -- cartesian product
    cart s = S.fromList $ do
      x <- S.toList s
      y <- S.toList s
      return (x, y)


-- | Run the network over a graph labeled with input word embeddings.
run
  :: ( KnownNat d
    , Ord alb, Show alb
    , Ord nlb, Show nlb
    , Reifies s W
     )
  => BVar s (Param d alb nlb)
    -- ^ Graph network / params
  -> Graph (Node d nlb) alb
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s Double)
    -- ^ Output map with output potential values
run net graph = M.fromList $ do
  (v, w) <- graphArcs graph
  let nodeMap = fmap (BP.constVar . nodeEmb) (nodeLabelMap graph)
      hv = nodeMap M.! v
      hw = nodeMap M.! w
      biPot = Just
            $ (net ^^. potR) `dot`
              (FFN.run (net ^^. potN) (hv # hw))
--       biAff = do
--         biaff <- BP.sequenceVar (net ^^. arcBiaff)
--         return $ biaff #> (hv # hw)
--       -- biAff = Just $ (net ^^. arcBiaff) #> (hv # hw)
--       unordBiAff = do
--         biaff <- BP.sequenceVar (net ^^. arcUnordBiaff)
--         vLex <- nodeLex <$> M.lookup v (nodeLabelMap graph)
--         wLex <- nodeLex <$> M.lookup w (nodeLabelMap graph)
--         return $ if vLex <= wLex
--            then biaff #> (hv # hw)
--            else biaff #> (hw # hv)
--       srcAff = do
--         aff <- BP.sequenceVar (net ^^. sourceAff)
--         return $ aff #> hv
--       trgAff = do
--         aff <- BP.sequenceVar (net ^^. targetAff)
--         return $ aff #> hw
      arcBias = do
        arcBiasMap <- BP.sequenceVar (net ^^. arcLabelB)
        let err = trace
              ( "ArcGraph.run: unknown arc label ("
              ++ show (M.lookup (v, w) (arcLabelMap graph))
              ++ ")" ) 0
        return . maybe err id $ do
          arcLabel <- M.lookup (v, w) (arcLabelMap graph)
          arcBiasMap ^^? ixAt arcLabel
--       arcBias = do
--         arcBiasMap <- BP.sequenceVar (net ^^. arcLabelB)
--         let err = trace
--               ( "ArcGraph.run: unknown arc label ("
--               ++ show (M.lookup (v, w) (arcLabelMap graph))
--               ++ ")" ) 0
--         return . maybe err id $ do
--           arcLabel <- M.lookup (v, w) (arcLabelMap graph)
--           arcBiasMap ^^? ixAt arcLabel
--       nodePairBias = do
--         nodeBiasMap <- BP.sequenceVar (net ^^. nodeLabelB)
--         -- Construct the node labels here for the sake of error reporting
--         let nodeLabels = do
--               vLabel <- nodeLab <$> M.lookup v (nodeLabelMap graph)
--               wLabel <- nodeLab <$> M.lookup v (nodeLabelMap graph)
--               return (vLabel, wLabel)
--             err = trace
--               ( "ArcGraph.run: undefined node label(s) ("
--               ++ show nodeLabels
--               ++ ")" ) 0
--         return . maybe err id $ do
--           vLabel <- nodeLab <$> M.lookup v (nodeLabelMap graph)
--           wLabel <- nodeLab <$> M.lookup v (nodeLabelMap graph)
--           nodeBiasMap ^^? ixAt (vLabel, wLabel)
      bias = Just $ net ^^. arcB
--       he = sum . catMaybes $
--         [ biAff, unordBiAff, srcAff, trgAff
--         , arcBias, nodePairBias, bias
--         ]
--       -- x = softmax $ (net ^^. probM) #> he
--       x = softmax $ FFN.run (net ^^. probN) he
      x = sum . catMaybes $
        [ biPot, arcBias, bias
        ]
  return ((v, w), logistic x)
--     go prevNodeMap prevArcMap
--       | k <= 0 = M.fromList $ do
--           (p, q) <- graphArcs graph
--           let v = prevNodeMap M.! p
--               w = prevNodeMap M.! q
--           let x = (net ^^. potR) `dot`
--                   (FFN.run (net ^^. potN) (v # w))
--           return ((p, q), logistic x)


-- | Evaluate the network over a graph labeled with input word embeddings.
-- User-friendly (and without back-propagation) version of `run`.
eval
  :: ( KnownNat d
     , Ord alb, Show alb
     , Ord nlb, Show nlb )
  => Param d alb nlb
    -- ^ Graph network / params
  -- -> Graph (R d) b
  -> Graph (Node d nlb) alb
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc Double
eval net graph =
  BP.evalBP0
    ( BP.collectVar
    $ run
        (BP.constVar net)
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
-- Parameters
----------------------------------------------


type NewParam dim a b
   = Biaff dim dim
  :& ArcBias b
  :& Bias


run' 
  :: (KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W)
  => BVar s (NewParam dim a b)
    -- ^ Graph network / params
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s Double)
    -- ^ Output map with output potential values
run' net graph = M.fromList $ do
  arc <- graphArcs graph
  let x = runBiComp graph arc net
  return (arc, logistic x)


----------------------------------------------
-- Serialization
----------------------------------------------


-- | Save the parameters in the given file.
saveParam
  :: (KnownNat d, Binary alb, Binary nlb)
  => FilePath
  -> Param d alb nlb
  -> IO ()
saveParam path =
  B.writeFile path . compress . Bin.encode


-- | Load the parameters from the given file.
loadParam
  :: (KnownNat d, Binary alb, Binary nlb)
  => FilePath
  -> IO (Param d alb nlb)
loadParam path =
  Bin.decode . decompress <$> B.readFile path
  -- B.writeFile path . compress . Bin.encode


----------------------------------------------
-- DataSet
----------------------------------------------


-- | Dataset element
data Elem dim arc nlb = Elem
  { graph :: Graph (Node dim nlb) arc
    -- ^ Input graph
  , labMap :: M.Map Arc Double
    -- ^ Target probabilities
  } deriving (Show, Generic, Binary)


-- | DataSet: a list of (graph, target value map) pairs.
type DataSet d alb nlb = [Elem d alb nlb]


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
  :: (Reifies s W)
  => BVar s Double
    -- ^ Target ,,true'' MWE probability
  -> BVar s Double
    -- ^ Output ,,artificial'' MWE probability
  -> BVar s Double
crossEntropy p q = negate
  ( p0 * log q0
  + p1 * log q1
  )
  where
    p1 = p
    p0 = 1 - p1
    q1 = q
    q0 = 1 - q1


-- -- | Cross entropy between the true and the artificial distributions
-- crossEntropy
--   :: (KnownNat n, Reifies s W)
--   => BVar s (R n)
--     -- ^ Target ,,true'' distribution
--   -> BVar s (R n)
--     -- ^ Output ,,artificial'' distribution
--   -> BVar s Double
-- crossEntropy p q =
--   negate (p `dot` LBP.vmap' log q)


-- | Cross entropy between the target and the output values
errorOne
  :: (Ord a, Reifies s W)
--   => M.Map a (BVar s (R n))
  => M.Map a (BVar s Double)
    -- ^ Target values
--   -> M.Map a (BVar s (R n))
  -> M.Map a (BVar s Double)
    -- ^ Output values
  -> BVar s Double
errorOne target output = PB.sum . BP.collectVar $ do
  (key, tval) <- M.toList target
  let oval = output M.! key
  return $ crossEntropy tval oval


-- | Error on a dataset.
errorMany
  :: (Ord a, Reifies s W)
--   => [M.Map a (BVar s (R n))] -- ^ Targets
--   -> [M.Map a (BVar s (R n))] -- ^ Outputs
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
  :: ( Reifies s W, KnownNat d
     , Ord alb, Show alb, Ord nlb, Show nlb
     )
  => DataSet d alb nlb
  -> BVar s (Param d alb nlb)
  -> BVar s Double
netError dataSet net =
  let
    inputs = map graph dataSet
    -- outputs = map (run net . nmap BP.auto) inputs
    outputs = map (run net) inputs
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
