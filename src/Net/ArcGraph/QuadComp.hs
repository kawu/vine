{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MultiParamTypeClasses #-}
-- {-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveAnyClass #-}
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


module Net.ArcGraph.QuadComp
  ( QuadComp (..)
  , BiQuad (..)

  , QuadAff (..)
  , TriAff (..)
  , SibAff (..)
  , BiAff (..)
  , UnAff (..)
  , Bias (..)
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat, natVal)
import qualified GHC.TypeNats as Nats

import           Control.DeepSeq (NFData)
import           Control.Lens.At (ixAt)
import           Control.Monad (forM)

import           Lens.Micro.TH (makeLenses)

import           Data.Proxy (Proxy(..))
import qualified Data.Graph as G
import           Data.Binary (Binary)
import           Data.Maybe (mapMaybe)
import qualified Data.Set as S
import qualified Data.Map.Strict as M

import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, Reifies, W, BVar, (^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop (R, dot, (#))

import           Numeric.SGD.ParamSet (ParamSet)

import           Net.Basic
import           Net.ArcGraph.Graph
import qualified Net.ArcGraph.BiComp as B
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))

import           Debug.Trace (trace)


----------------------------------------------
-- Graph utils
----------------------------------------------


-- | Retrieve the preceding sister node of the given node w.r.t. the given
-- parent.
--
-- TODO (CRITICAL!): does the graph actually preserve the order of nodes?!  If
-- not, this is all wrong!  Therefore, you really should take some time in
-- order to make sure (with unit tests?) that this is correct.
--
-- In particular, you could use QuickCheck / SmallCheck to make sure that the
-- order of arcs in the created graph is consistent with token IDs (either the
-- same or inversed).
--
-- For the moment, it looks like:
-- 
--   * The graph is built (using `G.buildG`) from the list of arcs given in the
--     sequential order (i.e., as in the input sentence)
--   * The provided order is either preserved or reversed (but no permutation
--     or shuffling takes place)
--
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
    runQuadComp graph arc comp1 `M.union` runQuadComp graph arc comp2

-- instance (Backprop comp, B.BiComp dim a b comp) => QuadComp dim a b comp where
--   runQuadComp graph arc biComp =
--     M.singleton arc (B.runBiComp graph arc biComp)


newtype BiQuad comp = BiQuad
  { _unBiQuad :: comp
  } deriving (Show, Eq, Ord, Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''BiQuad

instance B.BiComp dim a b comp => QuadComp dim a b (BiQuad comp) where
  runQuadComp graph arc biQuad =
    let biComp = biQuad ^^. unBiQuad
     in M.singleton arc (B.runBiComp graph arc biComp)


----------------------------------------------
-- Components
----------------------------------------------


-- | Global bias
newtype Bias = Bias
  { _biasVal :: Double
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''Bias
instance New a b Bias where
  new xs ys = Bias <$> new xs ys
instance QuadComp dim a b Bias where
  runQuadComp _ arc bias =
    M.singleton arc (bias ^^. biasVal)


----------------------------------------------
----------------------------------------------


data UnAff d h = UnAff
  { _headAffN :: FFN d h 2
  , _depAffN  :: FFN d h 2
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''UnAff

instance (KnownNat d, KnownNat h) => New a b (UnAff d h) where
  new _ _ = UnAff
    <$> FFN.new d h 2
    <*> FFN.new d h 2
    where
      d = proxyVal (Proxy :: Proxy d)
      h = proxyVal (Proxy :: Proxy h)
      proxyVal = fromInteger . toInteger . natVal

instance (KnownNat d, KnownNat h) => QuadComp d a b (UnAff d h) where
  runQuadComp graph (v, w) una =
    let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        hv = wordRepr v
        hw = wordRepr w
        vecv = FFN.run (una ^^.  depAffN) hv
        vecw = FFN.run (una ^^. headAffN) hw
     in M.singleton (v, w) (elem0 vecv + elem0 vecw)


----------------------------------------------
----------------------------------------------


-- | Bi-affinity component, capturing the word embeddings of the current word and
-- its parent.
--
-- TODO: the output size should be @1@!
--
data BiAff d h = BiAff
  { _biAffN :: FFN
      (d Nats.+ d)
      h
      2
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''BiAff

instance (KnownNat d, KnownNat h) => New a b (BiAff d h) where
  new _ _ = BiAff
    <$> FFN.new (d*2) h 2
    where
      d = proxyVal (Proxy :: Proxy d)
      h = proxyVal (Proxy :: Proxy h)
      proxyVal = fromInteger . toInteger . natVal

instance (KnownNat d, KnownNat h) => QuadComp d a b (BiAff d h) where
  runQuadComp graph (v, w) bi =
    let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        hv = wordRepr v
        hw = wordRepr w
        vec = FFN.run (bi ^^. biAffN) (hv # hw)
     in M.singleton (v, w) (elem0 vec)


----------------------------------------------
----------------------------------------------


-- | Triple-affinity component, capturing the word embeddings of the current word,
-- its parent, and its grandparent.
--
-- TODO: the output size should be @2@!
--
data TriAff d h = TriAff
  { _triAffN :: FFN
      (d Nats.+ d Nats.+ d)
      h
      3
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''TriAff

instance (KnownNat d, KnownNat h) => New a b (TriAff d h) where
  new _ _ = TriAff
    <$> FFN.new (d*3) h 3
    where
      d = proxyVal (Proxy :: Proxy d)
      h = proxyVal (Proxy :: Proxy h)
      proxyVal = fromInteger . toInteger . natVal

instance (KnownNat d, KnownNat h) => QuadComp d a b (TriAff d h) where
  runQuadComp graph (v, w) tri =
    case outgoing w graph of
      [u] ->
        let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
            hv = wordRepr v
            hw = wordRepr w
            hu = wordRepr u
            vec = FFN.run (tri ^^. triAffN) (hv # hw # hu)
         in M.fromList
              [ ((v, w), elem0 vec)
              , ((w, u), elem1 vec)
              ]
      _ -> M.empty


----------------------------------------------
----------------------------------------------


-- | Sibling-affinity component, capturing the word embeddings of the current
-- word, its parent, and its sibling.
--
-- TODO: the output size should be @2@!
--
data SibAff d h = SibAff
  { _sibAffN :: FFN
      (d Nats.+ d Nats.+ d)
      h
      3
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''SibAff

instance (KnownNat d, KnownNat h) => New a b (SibAff d h) where
  new _ _ = SibAff
    <$> FFN.new (d*3) h 3
    where
      d = proxyVal (Proxy :: Proxy d)
      h = proxyVal (Proxy :: Proxy h)
      proxyVal = fromInteger . toInteger . natVal

instance (KnownNat d, KnownNat h) => QuadComp d a b (SibAff d h) where
  runQuadComp graph (v, w) sib = maybe M.empty id $ do
    s <- sister v w graph
    let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        hv = wordRepr v
        hw = wordRepr w
        hs = wordRepr s
        vec = FFN.run (sib ^^. sibAffN) (hv # hw # hs)
    return $ M.fromList
      [ ((v, w), elem0 vec)
      , ((s, w), elem1 vec)
      ]


----------------------------------------------
----------------------------------------------


-- | Quad-affinity component.
--
-- TODO: the output size should be @3@!
--
data QuadAff d h = QuadAff
  { _quadAffN :: FFN
      (d Nats.+ d Nats.+ d Nats.+ d)
      h
      4
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''QuadAff

instance (KnownNat d, KnownNat h) => New a b (QuadAff d h) where
  new _ _ = QuadAff
    <$> FFN.new (d*4) h 4
    where
      d = proxyVal (Proxy :: Proxy d)
      h = proxyVal (Proxy :: Proxy h)
      proxyVal = fromInteger . toInteger . natVal

instance (KnownNat d, KnownNat h) => QuadComp d a b (QuadAff d h) where
  runQuadComp graph (v, w) quad = maybe M.empty id $ do
    [u] <- return (outgoing w graph)
    s <- sister v w graph
    let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        hv = wordRepr v
        hw = wordRepr w
        hu = wordRepr u
        hs = wordRepr s
        vec = FFN.run (quad ^^. quadAffN) (hv # hw # hu # hs)
    return $ M.fromList
      [ ((v, w), elem0 vec)
      , ((w, u), elem1 vec)
      , ((s, w), elem2 vec)
      ]
