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

{-# LANGUAGE PolyKinds #-}

{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE Rank2Types #-}

{-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- To derive Binary for `Param`
{-# LANGUAGE DeriveAnyClass #-}

-- To make GHC automatically infer that `KnownNat d => KnownNat (d + d)`
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}


module Net.ArcGraph
  ( 
  -- * Network
--     Config(..)
    Param
  , printParam
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
import           GHC.TypeNats (KnownNat, natVal)
-- import           GHC.Natural (naturalToInt)
import qualified GHC.TypeNats as Nats

import           Control.Monad (forM_, forM)

import           Lens.Micro.TH (makeLenses)
-- import           Lens.Micro ((^.))

import           Data.Proxy (Proxy(..))
-- import           Data.Ord (comparing)
-- import qualified Data.List as L
import           Data.Maybe (catMaybes, mapMaybe)
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


-- | POS tag of the given vertex. 
posTag :: Graph (Node d a) b -> G.Vertex -> Maybe a
posTag g v = nodeLab <$> M.lookup v (nodeLabelMap g)


-- -- | Arc label of the given arc. 
-- arcLabel :: Graph (Node d a) b -> Arc -> Maybe a
-- arcLabel g (v, w) = M.lookup v (arcLabelMap g)


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
-- Various Utils
----------------------------------------------


-- | Remove duplicates.
nub :: (Ord a) => [a] -> [a]
nub = S.toList . S.fromList


----------------------------------------------
-- Config
----------------------------------------------
-- 
-- 
-- -- | Network configuration
-- data Config = Config
--   { useWordAff :: Bool
--     -- ^ Use word affinities
--   , useArcLabels :: Bool
--     -- ^ Use arc labels (UD dep rels)
--   , useNodeLabels :: Bool
--     -- ^ Use node labels (POS tags)
--   , useBiaff :: Bool
--     -- ^ Use ,,biaffine bias''
--   , useUnordBiaff :: Bool
--     -- ^ Use ,,unordered'' biaffinity.  The ,,standard'' biaffinity captures
--     -- the bias of an ordered pair of words (represented by their embeddings),
--     -- with the order determined by the direction of the dependency relation
--     -- between them.  The ,,unordered'' version is not really unordered, it's
--     -- rather that the order between two words is determined lexicographically
--     -- (based on `nodeLex`).  This allows to capture MWEs occurrences with a
--     -- reversed dependency direction, e.g., in the passive voice.
--   } deriving (Generic)
-- 
-- instance JSON.FromJSON Config
-- instance JSON.ToJSON Config where
--   toEncoding = JSON.genericToEncoding JSON.defaultOptions
-- instance Interpret Config
-- 
-- 
----------------------------------------------
-- Type voodoo
----------------------------------------------


-- | Custom pair, with Backprop and ParamSet instances, and nice Backprop
-- pattern.
data a :& b = !a :& !b
  deriving (Show, Generic, Binary, NFData, Backprop, ParamSet)
infixr 2 :&

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
  new
    :: S.Set a
      -- ^ Set of node labels
    -> S.Set b
      -- ^ Set of arc labels
    -> IO p

instance (New a b p1, New a b p2) => New a b (p1 :& p2) where
  new xs ys = do
    p1 <- new xs ys
    p2 <- new xs ys
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


-- -- | Activable component
-- data Activable a = Activable
--   { _actElem :: a
--   , _active :: Double
--     -- ^ Active if @active > 0@
--   } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
-- makeLenses ''Activable
-- instance (New a b c) => New a b (Activable c) where
--   new xs ys = Activable <$> new xs ys <*> pure 1
-- instance BiComp dim a b c => BiComp dim a b (Activable c) where
--   runBiComp graph arc act = 
--     if _active act > 0
--        then act ^^. actElem
--        else 0


----------------------------------------------
----------------------------------------------


-- | Global bias
newtype Bias = Bias
  { _biasVal :: Double
  } deriving (Show, Generic, Binary, NFData, ParamSet)
instance Backprop Bias
makeLenses ''Bias
instance New a b Bias where
  new _ _ = pure (Bias 0)
instance BiComp dim a b Bias where
  runBiComp _ _ bias = bias ^^. biasVal


----------------------------------------------
----------------------------------------------


-- | Arc label bias
newtype ArcBias b = ArcBias
  { _arcBiasMap :: M.Map b Double
  } deriving (Show, Generic, Binary, NFData, ParamSet)

instance (Ord b) => Backprop (ArcBias b)
makeLenses ''ArcBias

instance (Ord b) => New a b (ArcBias b) where
  new _ arcLabelSet =
    -- TODO: could be simplified...
    ArcBias <$> mkMap arcLabelSet (pure 0)
    where
      mkMap keySet mkVal = fmap M.fromList .
        forM (S.toList keySet) $ \key -> do
          (key,) <$> mkVal


-- | Label affinity of the given arc
arcAff
  :: (Ord a, Show a, Reifies s W)
  => a -> BVar s (ArcBias a) -> BVar s Double
arcAff x aff = maybe err id $ do
  aff ^^. arcBiasMap ^^? ixAt x
  where
    err = trace
      ( "ArcGraph.arcAff: unknown arc label ("
      ++ show x
      ++ ")" ) 0

instance (Ord b, Show b) => BiComp dim a b (ArcBias b) where
  runBiComp graph (v, w) arcBias =
    let err = trace
          ( "ArcGraph.run: unknown arc ("
          ++ show (v, w)
          ++ ")" ) 0
     in maybe err id $ do
          arcLabel <- M.lookup (v, w) (arcLabelMap graph)
          return $ arcAff arcLabel arcBias


-- | Parent/grandparent arc label affinity
data PapyArcAff b = PapyArcAff
  { _papyArcAff :: ArcBias b
  , _papyArcDef :: Double
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''PapyArcAff
instance (Ord b) => New a b (PapyArcAff b) where
  new xs ys = PapyArcAff <$> new xs ys <*> pure 0
instance (Ord b, Show b) => BiComp dim a b (PapyArcAff b) where
  runBiComp graph (_, w) aff = check $ do
    x <- nub . mapMaybe (arcLabel.(w,)) $ outgoing w graph
    return $ arcAff x (aff ^^. papyArcAff)
    where
      arcLabel arc = M.lookup arc (arcLabelMap graph)
      check [] = aff ^^. papyArcDef
      check xs = sum xs


-- | Child/grandchild arc label affinity
data EnkelArcAff b = EnkelArcAff
  { _enkelnArcAff :: ArcBias b
  , _enkelnArcDef :: Double
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''EnkelArcAff
instance (Ord b) => New a b (EnkelArcAff b) where
  new xs ys = EnkelArcAff <$> new xs ys <*> pure 0
instance (Ord b, Show b) => BiComp dim a b (EnkelArcAff b) where
  runBiComp graph (v, _) aff = check $ do
    x <- nub . mapMaybe (arcLabel.(,v)) $ incoming v graph
    return $ arcAff x (aff ^^. enkelnArcAff)
    where
      arcLabel arc = M.lookup arc (arcLabelMap graph)
      check [] = aff ^^. enkelnArcDef
      check xs = sum xs


----------------------------------------------
----------------------------------------------


-- | Word affinity component
data Aff d h = Aff
  { _affN :: FFN
      d
      h -- Hidden layer size
      d -- Output
    -- ^ Potential FFN
  , _affV :: R d
    -- ^ Potential ,,feature vector''
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat dim, KnownNat h) => Backprop (Aff dim h)
makeLenses ''Aff

instance (KnownNat dim, KnownNat h) => New a b (Aff dim h) where
  new _ _ = Aff
    <$> FFN.new d h d
    <*> vector d
    where
      d = proxyVal (Proxy :: Proxy dim)
      h = proxyVal (Proxy :: Proxy h)
      proxyVal = fromInteger . toInteger . natVal


-- | Affinity of the given node
nodeAff :: (KnownNat d, KnownNat h, Reifies s W)
        => Graph (Node d a) b -> G.Vertex -> BVar s (Aff d h) -> BVar s Double
nodeAff graph v aff =
  let nodeMap = fmap
        (BP.constVar . nodeEmb)
        (nodeLabelMap graph)
      hv = nodeMap M.! v
   in (aff ^^. affV) `dot`
        (FFN.run (aff ^^. affN) hv)


newtype HeadAff d h = HeadAff { _unHeadAff :: Aff d h }
  deriving (Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''HeadAff
instance (KnownNat dim, KnownNat h) => New a b (HeadAff dim h) where
  new xs ys = HeadAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim a b (HeadAff dim h) where
  runBiComp graph (_, w) aff = nodeAff graph w (aff ^^. unHeadAff)


newtype DepAff d h = DepAff { _unDepAff :: Aff d h }
  deriving (Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''DepAff
instance (KnownNat dim, KnownNat h) => New a b (DepAff dim h) where
  new xs ys = DepAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim a b (DepAff dim h) where
  runBiComp graph (v, _) aff = nodeAff graph v (aff ^^. unDepAff)


newtype SymAff d h = SymAff { _unSymAff :: Aff d h }
  deriving (Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''SymAff
instance (KnownNat dim, KnownNat h) => New a b (SymAff dim h) where
  new xs ys = SymAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim a b (SymAff dim h) where
  runBiComp graph (v, w) aff
    = nodeAff graph v (aff ^^. unSymAff)
    + nodeAff graph w (aff ^^. unSymAff)


-- instance (KnownNat dim, KnownNat h) => BiComp dim a b (Aff dim h) where
--   runBiComp graph (v, w) aff =
--      let nodeMap = fmap
--            (BP.constVar . nodeEmb)
--            (nodeLabelMap graph)
--          hv = nodeMap M.! v
--          hw = nodeMap M.! w
--          pot1 = (aff ^^. affV) `dot`
--            (FFN.run (aff ^^. affN) hv)
--          pot2 = (aff ^^. affV) `dot`
--            (FFN.run (aff ^^. affN) hw)
--       in pot1 + pot2


----------------------------------------------
----------------------------------------------


-- | POS affinity component
data Pos a = Pos
  { _posMap :: M.Map a Double
  } deriving (Show, Generic, Binary, NFData, ParamSet)

instance (Ord a) => Backprop (Pos a)
makeLenses ''Pos

instance (Ord a) => New a b (Pos a) where
  new nodeLabelSet _ =
    Pos <$> mkMap nodeLabelSet (pure 0)
    where
      mkMap keySet mkVal = fmap M.fromList .
        forM (S.toList keySet) $ \key -> do
          (key,) <$> mkVal


-- | POS affinity of the given node
posAff
  :: (Ord a, Show a, Reifies s W)
  => a -> BVar s (Pos a) -> BVar s Double
posAff pos aff = maybe err id $ do
  aff ^^. posMap ^^? ixAt pos
  where
    err = trace
      ( "ArcGraph.posAff: unknown POS ("
      ++ show pos
      ++ ")" ) 0


-- | POS affinity of the given node
nodePosAff
  :: (Ord a, Show a, Reifies s W)
  => Graph (Node d a) b -> G.Vertex
  -> BVar s (Pos a) -> BVar s Double
nodePosAff graph v aff = maybe err id $ do
  pos <- nodeLab <$> M.lookup v (nodeLabelMap graph)
  return (posAff pos aff)
  where
    err = trace
      ( "ArcGraph.nodePosAff: undefined node ("
      ++ show v
      ++ ")" ) 0


newtype HeadPosAff a = HeadPosAff { _unHeadPosAff :: Pos a }
  deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''HeadPosAff
instance (Ord a) => New a b (HeadPosAff a) where
  new xs ys = HeadPosAff <$> new xs ys
instance (Ord a, Show a) => BiComp dim a b (HeadPosAff a) where
  runBiComp graph (_, w) aff = nodePosAff graph w (aff ^^. unHeadPosAff)


newtype DepPosAff a = DepPosAff { _unDepPosAff :: Pos a }
  deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''DepPosAff
instance (Ord a) => New a b (DepPosAff a) where
  new xs ys = DepPosAff <$> new xs ys
instance (Ord a, Show a) => BiComp dim a b (DepPosAff a) where
  runBiComp graph (v, _) aff = nodePosAff graph v (aff ^^. unDepPosAff)


data PapyPosAff a = PapyPosAff
  { _papyPosAff :: Pos a 
  , _papyPosDef :: Double
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''PapyPosAff
instance (Ord a) => New a b (PapyPosAff a) where
  new xs ys = PapyPosAff <$> new xs ys <*> pure 0
instance (Ord a, Show a) => BiComp dim a b (PapyPosAff a) where
  runBiComp graph (_, w) aff = check $ do
    pos <- nub . mapMaybe (posTag graph) $ outgoing w graph
    return $ posAff pos (aff ^^. papyPosAff)
    where
      check [] = aff ^^. papyPosDef
      check xs = sum xs


data EnkelPosAff a = EnkelPosAff
  { _enkelPosAff :: Pos a 
  , _enkelPosDef :: Double
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''EnkelPosAff
instance (Ord a) => New a b (EnkelPosAff a) where
  new xs ys = EnkelPosAff <$> new xs ys <*> pure 0
instance (Ord a, Show a) => BiComp dim a b (EnkelPosAff a) where
  runBiComp graph (v, _) aff = check $ do
    pos <- nub . mapMaybe (posTag graph) $ incoming v graph
    return $ posAff pos (aff ^^. enkelPosAff)
    where
      check [] = aff ^^. enkelPosDef
      check xs = sum xs


-- newtype DepAff d h = DepAff { _unDepAff :: Aff d h }
--   deriving (Generic, Binary, NFData, ParamSet, Backprop)
-- makeLenses ''DepAff
-- instance (KnownNat dim, KnownNat h) => New a b (DepAff dim h) where
--   new xs ys = DepAff <$> new xs ys
-- instance (KnownNat dim, KnownNat h) => BiComp dim a b (DepAff dim h) where
--   runBiComp graph (v, _) aff = nodeAff graph v (aff ^^. unDepAff)
-- 
-- 
-- newtype SymAff d h = SymAff { _unSymAff :: Aff d h }
--   deriving (Generic, Binary, NFData, ParamSet, Backprop)
-- makeLenses ''SymAff
-- instance (KnownNat dim, KnownNat h) => New a b (SymAff dim h) where
--   new xs ys = SymAff <$> new xs ys
-- instance (KnownNat dim, KnownNat h) => BiComp dim a b (SymAff dim h) where
--   runBiComp graph (v, w) aff
--     = nodeAff graph v (aff ^^. unSymAff)
--     + nodeAff graph w (aff ^^. unSymAff)


----------------------------------------------
----------------------------------------------


-- -- | Head affinity component
-- data HeadAff d h = HeadAff
--   { _affN :: FFN
--       d
--       h -- Hidden layer size
--       d -- Output
--     -- ^ Potential FFN
--   , _affV :: R d
--     -- ^ Potential ,,feature vector''
--   } deriving (Generic, Binary, NFData, ParamSet)
-- 
-- instance (KnownNat dim, KnownNat h) => Backprop (HeadAff dim h)
-- makeLenses ''HeadAff
-- 
-- instance (KnownNat dim, KnownNat h) => New a b (HeadAff dim h) where
--   new _ _ = HeadAff
--     <$> FFN.new d h d
--     <*> vector d
--     where
--       d = proxyVal (Proxy :: Proxy dim)
--       h = proxyVal (Proxy :: Proxy h)
--       proxyVal = fromInteger . toInteger . natVal
-- 
-- instance (KnownNat dim, KnownNat h) => BiComp dim a b (HeadAff dim h) where
--   runBiComp graph (v, w) aff =
--      let nodeMap = fmap
--            (BP.constVar . nodeEmb)
--            (nodeLabelMap graph)
--          hv = nodeMap M.! v
--          hw = nodeMap M.! w
--          pot1 = (aff ^^. affV) `dot`
--            (FFN.run (aff ^^. affN) hv)
--          pot2 = (aff ^^. affV) `dot`
--            (FFN.run (aff ^^. affN) hw)
--       in pot1 + pot2


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
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat dim, KnownNat h) => Backprop (Biaff dim h)
makeLenses ''Biaff

instance (KnownNat dim, KnownNat h) => New a b (Biaff dim h) where
  new _ _ = Biaff
    <$> FFN.new (d*2) h d
    <*> vector d
    where
      d = proxyVal (Proxy :: Proxy dim)
      h = proxyVal (Proxy :: Proxy h)
      proxyVal = fromInteger . toInteger . natVal

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
----------------------------------------------


-- | Unordered biaffinity component
data UnordBiaff d h = UnordBiaff
  { _unordBiaffN :: FFN
      (d Nats.+ d)
      h -- Hidden layer size
      d -- Output
    -- ^ Potential FFN
  , _unordBiaffV :: R d
    -- ^ Potential ,,feature vector''
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat dim, KnownNat h) => Backprop (UnordBiaff dim h)
makeLenses ''UnordBiaff

instance (KnownNat dim, KnownNat h) => New a b (UnordBiaff dim h) where
  new _ _ = UnordBiaff
    <$> FFN.new (d*2) h d
    <*> vector d
    where
      d = proxyVal (Proxy :: Proxy dim)
      h = proxyVal (Proxy :: Proxy h)
      proxyVal = fromInteger . toInteger . natVal

instance (KnownNat dim, KnownNat h) => BiComp dim a b (UnordBiaff dim h) where
  runBiComp graph (v, w) unordBia =
     let vLex = nodeLex <$> M.lookup v (nodeLabelMap graph)
         wLex = nodeLex <$> M.lookup w (nodeLabelMap graph)
         embMap = fmap
           (BP.constVar . nodeEmb)
           (nodeLabelMap graph)
         emb = (embMap M.!)
         (hv, hw) =
           if vLex <= wLex
              then (emb v, emb w)
              else (emb w, emb v)
      in (unordBia ^^. unordBiaffV) `dot`
           (FFN.run (unordBia ^^. unordBiaffN) (hv # hw))


----------------------------------------------
----------------------------------------------


-- | A version of `Biaff` in which the label between two nodes determines who
-- is the ,,semantic'' parent.
data DirBiaff dim h b = DirBiaff
  { _dirBiaMap :: M.Map b Double
  , _dirBiaff :: Biaff dim h
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat dim, KnownNat h, Ord b) => Backprop (DirBiaff dim h b)
makeLenses ''DirBiaff

instance (KnownNat dim, KnownNat h, Ord b) => New a b (DirBiaff dim h b) where
  new nodeLabelSet arcLabelSet = DirBiaff
    -- TODO: could be simplified...
    <$> mkMap arcLabelSet (pure 0)
    <*> new nodeLabelSet arcLabelSet
    where
      mkMap keySet mkVal = fmap M.fromList .
        forM (S.toList keySet) $ \key -> do
          (key,) <$> mkVal

instance (KnownNat dim, KnownNat h, Ord b, Show b)
  => BiComp dim a b (DirBiaff dim h b) where
  runBiComp graph (v, w) dirBia =
    let nodeMap = fmap
          (BP.constVar . nodeEmb)
          (nodeLabelMap graph)
        hv0 = nodeMap M.! v
        hw0 = nodeMap M.! w
        err = trace
          ( "ArcGraph.DirBiaff: unknown arc label ("
          ++ show (M.lookup (v, w) (arcLabelMap graph))
          ++ ")" ) 0
        p = maybe err logistic $ do
              arcLabel <- M.lookup (v, w) (arcLabelMap graph)
              dirBia ^^. dirBiaMap ^^? ixAt arcLabel
        q = 1 - p
        scale x = LBP.vmap (*x)
        hv = scale p hv0 + scale q hw0
        hw = scale q hv0 + scale p hw0
        bia = dirBia ^^. dirBiaff
     in (bia ^^. biaffV) `dot`
          (FFN.run (bia ^^. biaffN) (hv # hw))


----------------------------------------------
----------------------------------------------


-- -- | A version of `Biaff` in which the label between two nodes determines who
-- -- is the ,,semantic'' parent.
-- data DirBiaff dim h b = DirBiaff
--   { _dirBiaMap :: M.Map b Double
--   , _dirBiaff :: Biaff dim h
--   } deriving (Generic, Binary, NFData, ParamSet)
-- 
-- instance (KnownNat dim, KnownNat h, Ord b) => Backprop (DirBiaff dim h b)
-- makeLenses ''DirBiaff
-- 
-- instance (KnownNat dim, KnownNat h, Ord b) => New a b (DirBiaff dim h b) where
--   new nodeLabelSet arcLabelSet = DirBiaff
--     -- TODO: could be simplified...
--     <$> mkMap arcLabelSet (pure 0)
--     <*> new nodeLabelSet arcLabelSet
--     where
--       mkMap keySet mkVal = fmap M.fromList .
--         forM (S.toList keySet) $ \key -> do
--           (key,) <$> mkVal
-- 
-- instance (KnownNat dim, KnownNat h, Ord b, Show b)
--   => BiComp dim a b (DirBiaff dim h b) where
--   runBiComp graph (v, w) dirBia =
--     let nodeMap = fmap
--           (BP.constVar . nodeEmb)
--           (nodeLabelMap graph)
--         hv = nodeMap M.! v
--         hw = nodeMap M.! w
--         err = trace
--           ( "ArcGraph.DirBiaff: unknown arc label ("
--           ++ show (M.lookup (v, w) (arcLabelMap graph))
--           ++ ")" ) 1
--         -- logist x = 1.0 / (1.0 + exp (-100*x))
--         logist = logistic
--         p = maybe err logist $ do
--               arcLabel <- M.lookup (v, w) (arcLabelMap graph)
--               dirBia ^^. dirBiaMap ^^? ixAt arcLabel
--         q = 1 - p
--         bia = dirBia ^^. dirBiaff
--         pot1 = (bia ^^. biaffV) `dot`
--                   (FFN.run (bia ^^. biaffN) (hv # hw))
--         pot2 = (bia ^^. biaffV) `dot`
--                   (FFN.run (bia ^^. biaffN) (hw # hv))
--      in if p > 0.99
--         then p*pot1
--         else if q > 0.99
--         then q*pot2
--         else p*pot1 + q*pot2


----------------------------------------------
-- Parameters
----------------------------------------------


-- | Parameter set
type Param dim a b 
   = Biaff dim dim
  :& UnordBiaff dim dim
  :& HeadAff dim dim
  :& DepAff dim dim
  :& HeadPosAff a
  :& DepPosAff a
  :& PapyPosAff a
  :& EnkelPosAff a
  :& ArcBias b
  :& PapyArcAff b
  :& EnkelArcAff b
  :& Bias


printParam :: (Show a, Show b) => Param dim a b -> IO ()
printParam
  ( biaff :& unordBiaff :& headAff :& depAff :&
    headPosAff :& depPosAff :& papyPosAff :&
    enkelPosAff :& arcBias :& papyArcAff :&
    enkelArcAff :& bias
  ) = do
    print headPosAff
    print depPosAff
    print papyPosAff
    print enkelPosAff
    print arcBias
    print papyArcAff
    print enkelArcAff
    print bias


run
  :: (KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W)
  => BVar s (Param dim a b)
    -- ^ Graph network / params
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s Double)
    -- ^ Output map with output potential values
run net graph = M.fromList $ do
  arc <- graphArcs graph
  let x = runBiComp graph arc net
  return (arc, logistic x)


-- | Evaluate the network over a graph labeled with input word embeddings.
-- User-friendly (and without back-propagation) version of `run`.
eval
  :: (KnownNat dim, Ord a, Show a, Ord b, Show b)
  => Param dim a b
    -- ^ Graph network / params
  -- -> Graph (R d) b
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc Double
eval net graph =
  BP.evalBP0
    ( BP.collectVar
    $ run
        (BP.constVar net)
        graph
    )


----------------------------------------------
-- Serialization
----------------------------------------------


-- | Save the parameters in the given file.
saveParam :: (Binary a) => FilePath -> a -> IO ()
saveParam path =
  B.writeFile path . compress . Bin.encode


-- | Load the parameters from the given file.
loadParam :: (Binary a) => FilePath -> IO a
loadParam path =
  Bin.decode . decompress <$> B.readFile path
  -- B.writeFile path . compress . Bin.encode


----------------------------------------------
-- DataSet
----------------------------------------------


-- | Dataset element
data Elem dim a b = Elem
  { graph :: Graph (Node dim a) b
    -- ^ Input graph
  , labMap :: M.Map Arc Double
    -- ^ Target probabilities
  } deriving (Show, Generic, Binary)


-- | DataSet: a list of (graph, target value map) pairs.
type DataSet d a b = [Elem d a b]


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
  :: ( Reifies s W, KnownNat dim
     , Ord a, Show a, Ord b, Show b
     )
  => DataSet dim a b
  -> BVar s (Param dim a b)
  -> BVar s Double
netError dataSet net =
  let
    inputs = map graph dataSet
    -- outputs = map (run net . nmap BP.auto) inputs
    outputs = map (run net) inputs
    targets = map (fmap BP.auto . labMap) dataSet
  in  
    errorMany targets outputs -- + (size net * 0.01)
