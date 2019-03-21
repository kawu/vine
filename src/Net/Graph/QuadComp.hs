{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MultiParamTypeClasses #-}
-- {-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
-- {-# LANGUAGE PatternSynonyms #-}
-- {-# LANGUAGE LambdaCase #-}
-- {-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TupleSections #-}

-------------------------
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=5 #-}
-------------------------

-- {-# LANGUAGE UndecidableInstances #-}


module Net.Graph.QuadComp
  ( QuadComp (..)
  , BiQuad (..)

  , QuadAff (..)
  , TriAff (..)
  , SibAff (..)
  , BiAff (..)
  , BiAffExt (..)
  , UnordBiAff (..)
  , WordAff (..)
  , UnAff (..)
  , Bias (..)
  ) where


import           Prelude hiding (lex)

import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
import qualified GHC.TypeNats as Nats

import           Control.DeepSeq (NFData)
import           Control.Lens.At (ixAt, ix)
import qualified Control.Lens.At as At
-- import           Control.Monad (forM)

import           Lens.Micro.TH (makeLenses)

-- import           Data.Proxy (Proxy(..))
import qualified Data.Graph as G
import           Data.Binary (Binary)
-- import           Data.Maybe (mapMaybe)
-- import qualified Data.Set as S
import qualified Data.Map.Strict as M

-- import qualified Numeric.LinearAlgebra as LA
import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, Reifies, W, BVar, (^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop (R, (#))

import           Numeric.SGD.ParamSet (ParamSet)

import           Graph
import           Net.New
import           Net.Pair
import qualified Net.Graph.BiComp as B
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))

import           Debug.Trace (trace)


----------------------------------------------
-- Utils
----------------------------------------------


-- | Retrieve the preceding sister node of the given node w.r.t. the given
-- parent.
sister
  :: G.Vertex
    -- ^ Current vertex
  -> G.Vertex
    -- ^ Its parent
  -> Graph (Node dim a) b
    -- ^ The underlying graph
  -> Maybe G.Vertex
sister cur par graph =
  go Nothing (incoming par graph)
  where
    go _prev [] = Nothing
    go prev (x:xs)
      | x == cur = prev
      | otherwise = go (Just x) xs


at
  :: ( Num (At.IxValue b), Reifies s W, Backprop b
     , Backprop (At.IxValue b), At.Ixed b
     )
  => BVar s b
  -> At.Index b 
  -> BVar s (At.IxValue b)
at v k = maybe 0 id $ v ^^? ix k
{-# INLINE at #-}


----------------------------------------------
-- Quad component class
----------------------------------------------


-- | Quad component which captures the potential of (i) the reference arc, (ii)
-- the parent arc (if any), and (iii) the left sibling arc (if any).
--
-- At the moment the type of the result does not reflect this.  I.e., it's more
-- like a generic factor component with no constraints on the scope (it can for
-- instance return the potential of the grandparent arc).
--
class Backprop comp => QuadComp dim a b comp where
  runQuadComp
    :: (Reifies s W)
    => Graph (Node dim a) b
      -- ^ The underlying graph
    -> Arc
      -- ^ The reference arc
    -> BVar s comp
      -- ^ The component itself
    -> M.Map Arc (BVar s Double)
      -- ^ The resulting complex potential

instance (QuadComp dim a b comp1, QuadComp dim a b comp2)
  => QuadComp dim a b (comp1 :& comp2) where
  runQuadComp graph arc (comp1 :&& comp2) =
    M.unionWith (+)
      (runQuadComp graph arc comp1)
      (runQuadComp graph arc comp2)
    -- runQuadComp graph arc comp1 `M.union` runQuadComp graph arc comp2

-- instance (Backprop comp, B.BiComp dim a b comp) => QuadComp dim a b comp where
--   runQuadComp graph arc biComp =
--     M.singleton arc (B.runBiComp graph arc biComp)


newtype BiQuad comp = BiQuad
  { _unBiQuad :: comp
  } deriving (Show, Eq, Ord, Generic)
    deriving newtype (Binary, NFData, ParamSet, Backprop)

makeLenses ''BiQuad

instance B.BiComp dim a b comp => QuadComp dim a b (BiQuad comp) where
  runQuadComp graph arc biQuad =
    let biComp = biQuad ^^. unBiQuad
     in M.singleton arc (B.runBiComp graph arc biComp)

instance (New a b comp) => New a b (BiQuad comp) where
  new xs ys = BiQuad <$> new xs ys

----------------------------------------------
-- Components
----------------------------------------------


-- | Global bias
newtype Bias = Bias
  { _biasVal :: Double
  } deriving (Show, Generic)
    deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''Bias
instance New a b Bias where
  new xs ys = Bias <$> new xs ys
instance QuadComp dim a b Bias where
  runQuadComp _ arc bias =
    M.singleton arc (bias ^^. biasVal)


----------------------------------------------
----------------------------------------------


-- | Uni-word affinity component in which the head is not distinguished from
-- the dependent.  See also `UnAff`.  Note also that this could be implemented
-- more efficiently (as it is, the word-affinity of a word is re-calculated
-- several times if it is the head of several words).
newtype WordAff d h = WordAff
  { _wordAffN :: FFN d h 1
  } deriving (Generic)
    deriving newtype (Binary, NFData, ParamSet, Backprop)

makeLenses ''WordAff

instance (KnownNat d, KnownNat h) => New a b (WordAff d h) where
  new xs ys = WordAff <$> new xs ys

instance (KnownNat d, KnownNat h) => QuadComp d a b (WordAff d h) where
  runQuadComp graph (v, w) una =
    let emb = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        hv = emb v
        hw = emb w
        vecv = LBP.extractV $ FFN.run (una ^^. wordAffN) hv
        vecw = LBP.extractV $ FFN.run (una ^^. wordAffN) hw
     in M.singleton (v, w) (vecv `at` 0 + vecw `at` 0)


----------------------------------------------
----------------------------------------------


-- | Uni-word affinity component, with distinction between the head and the
-- dependent.  See also `WordAff`.
data UnAff d h = UnAff
  { _headAffN :: FFN d h 1
  , _depAffN  :: FFN d h 1
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''UnAff

instance (KnownNat d, KnownNat h) => New a b (UnAff d h) where
  new xs ys = UnAff
    <$> new xs ys
    <*> new xs ys

instance (KnownNat d, KnownNat h) => QuadComp d a b (UnAff d h) where
  runQuadComp graph (v, w) una =
    let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        hv = wordRepr v
        hw = wordRepr w
        -- vecv = FFN.run (una ^^.  depAffN) hv
        -- vecw = FFN.run (una ^^. headAffN) hw
        vecv = LBP.extractV $ FFN.run (una ^^.  depAffN) hv
        vecw = LBP.extractV $ FFN.run (una ^^. headAffN) hw
     in M.singleton (v, w) (vecv `at` 0 + vecw `at` 0)
     -- in M.singleton (v, w) (elem0 vecv + elem0 vecw)


----------------------------------------------
----------------------------------------------


-- | Bi-affinity component, capturing the word embeddings of the current word and
-- its parent.
data BiAff d h = BiAff
  { _biAffN :: FFN (d Nats.+ d) h 1
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''BiAff

instance (KnownNat d, KnownNat h) => New a b (BiAff d h) where
  new xs ys = BiAff <$> new xs ys

instance (KnownNat d, KnownNat h) => QuadComp d a b (BiAff d h) where
  runQuadComp graph (v, w) bi =
    let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        hv = wordRepr v
        hw = wordRepr w
        vec = LBP.extractV $ FFN.run (bi ^^. biAffN) (hv # hw)
     in M.singleton (v, w) (vec `at` 0)


----------------------------------------------
----------------------------------------------


-- | Biaffinity component extended with info about POS and DEP labels.
--
--   * @d@ -- word embedding dimension
--   * @l@ -- label (POS, DEP) embedding dimension
--   * @h@ -- hidden dimension
--
data BiAffExt d l a b h = BiAffExt
  { _biAffExtN :: FFN (d Nats.+ d Nats.+ l Nats.+ l Nats.+ l) h 1
    -- ^ The actual bi-affinity net
  , _biAffHeadPosMap :: M.Map a (R l)
    -- ^ Head POS repr
  , _biAffDepPosMap :: M.Map a (R l)
    -- ^ Dependent POS repr
  , _biAffArcMap :: M.Map b (R l)
    -- ^ Arc label repr
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''BiAffExt

instance (KnownNat d, KnownNat h, KnownNat l, Ord a, Ord b)
  => New a b (BiAffExt d l a b h) where
  new xs ys = BiAffExt
    <$> new xs ys
    <*> newMap xs xs ys
    <*> newMap xs xs ys
    <*> newMap ys xs ys

instance (KnownNat d, KnownNat h, KnownNat l, Ord a, Ord b, Show a, Show b)
  => QuadComp d a b (BiAffExt d l a b h) where
  runQuadComp graph (v, w) bi =
    let nodeMap = nodeLabelMap graph
        emb = BP.constVar . nodeEmb . (nodeMap M.!)
        depPos = depPosRepr . nodeLab $ nodeMap M.! v
        headPos = headPosRepr . nodeLab $ nodeMap M.! w
        arc = arcRepr $ arcLabelMap graph M.! (v, w)
        hv = emb v
        hw = emb w
        vec = LBP.extractV $ FFN.run (bi ^^. biAffExtN) 
          (hv # hw # depPos # headPos # arc)
     in M.singleton (v, w) (vec `at` 0)
    where
      headPosRepr pos = maybe err id $ do
        bi ^^. biAffHeadPosMap ^^? ixAt pos
        where
          err = trace
            ( "Graph.Holi: unknown POS ("
            ++ show pos
            ++ ")" ) 0
      depPosRepr pos = maybe err id $ do
        bi ^^. biAffDepPosMap ^^? ixAt pos
        where
          err = trace
            ( "Graph.Holi: unknown POS ("
            ++ show pos
            ++ ")" ) 0
      arcRepr dep = maybe err id $ do
        bi ^^. biAffArcMap ^^? ixAt dep
        where
          err = trace
            ( "Graph.Holi: unknown arc label ("
            ++ show dep
            ++ ")" ) 0


----------------------------------------------
----------------------------------------------


-- | Unordered bi-affinity component.
data UnordBiAff d h = UnordBiAff
  { _unordBiAffN :: FFN (d Nats.+ d) h 1
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''UnordBiAff

instance (KnownNat d, KnownNat h) => New a b (UnordBiAff d h) where
  new xs ys = UnordBiAff <$> new xs ys

instance (KnownNat d, KnownNat h) => QuadComp d a b (UnordBiAff d h) where
  runQuadComp graph (v, w) bi =
    let lex x = nodeLex <$> M.lookup x (nodeLabelMap graph)
        emb = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        (hv, hw) =
          if lex v <= lex w
             then (emb v, emb w)
             else (emb w, emb v)
        vec = LBP.extractV $ FFN.run (bi ^^. unordBiAffN) (hv # hw)
     in M.singleton (v, w) (vec `at` 0)


----------------------------------------------
----------------------------------------------


-- | Triple-affinity component, capturing the word embeddings of the current word,
-- its parent, and its grandparent.
data TriAff d h = TriAff
  { _triAffN :: FFN (d Nats.+ d Nats.+ d) h 2
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''TriAff

instance (KnownNat d, KnownNat h) => New a b (TriAff d h) where
  new xs ys = TriAff <$> new xs ys

instance (KnownNat d, KnownNat h) => QuadComp d a b (TriAff d h) where
  runQuadComp graph (v, w) tri = maybe M.empty id $ do
    [u] <- return (outgoing w graph)
    let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        hv = wordRepr v
        hw = wordRepr w
        hu = wordRepr u
        vec = LBP.extractV $ FFN.run (tri ^^. triAffN) (hv # hw # hu)
    return $  M.fromList
      [ ((v, w), vec `at` 0)
      , ((w, u), vec `at` 1)
      ]


----------------------------------------------
----------------------------------------------


-- | Sibling-affinity component, capturing the word embeddings of the current
-- word, its parent, and its sibling.
data SibAff d h = SibAff
  { _sibAffN :: FFN (d Nats.+ d Nats.+ d) h 2
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''SibAff

instance (KnownNat d, KnownNat h) => New a b (SibAff d h) where
  new xs ys = SibAff <$> new xs ys

instance (KnownNat d, KnownNat h) => QuadComp d a b (SibAff d h) where
  runQuadComp graph (v, w) sib = maybe M.empty id $ do
    s <- sister v w graph
    let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        hv = wordRepr v
        hw = wordRepr w
        hs = wordRepr s
        vec = LBP.extractV $ FFN.run (sib ^^. sibAffN) (hv # hw # hs)
    return $ M.fromList
      [ ((v, w), vec `at` 0)
      , ((s, w), vec `at` 1)
      ]


----------------------------------------------
----------------------------------------------


-- | Quad-affinity component.
data QuadAff d h = QuadAff
  { _quadAffN :: FFN (d Nats.+ d Nats.+ d Nats.+ d) h 3
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''QuadAff

instance (KnownNat d, KnownNat h) => New a b (QuadAff d h) where
  new xs ys = QuadAff <$> new xs ys

instance (KnownNat d, KnownNat h) => QuadComp d a b (QuadAff d h) where
  runQuadComp graph (v, w) quad = maybe M.empty id $ do
    [u] <- return (outgoing w graph)
    s <- sister v w graph
    let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        hv = wordRepr v
        hw = wordRepr w
        hu = wordRepr u
        hs = wordRepr s
        vec = LBP.extractV $ FFN.run (quad ^^. quadAffN) (hv # hw # hu # hs)
    return $ M.fromList
      [ ((v, w), vec `at` 0)
      , ((w, u), vec `at` 1)
      , ((s, w), vec `at` 2)
      ]
