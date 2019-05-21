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
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveFoldable #-}
-- {-# LANGUAGE DeriveTraversable #-}

-------------------------
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=5 #-}
-------------------------


module Net.Graph.BiComp
  ( BiComp (..)
  , Bias (..)

  , BiAff (..)
  , BiAffMix (..)

  , NoBi (..)
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
import qualified GHC.TypeNats as Nats

import           Control.DeepSeq (NFData)
import           Control.Lens.At (ixAt, ix)
import qualified Control.Lens.At as At
import           Control.Monad (forM)

import           Lens.Micro.TH (makeLenses)

-- import           Data.Proxy (Proxy(..))
import qualified Data.Graph as G
import           Data.Binary (Binary)
import           Data.Maybe (mapMaybe)
import qualified Data.Set as S
import qualified Data.Map.Strict as M

import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, Reifies, W, BVar, (^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop (R, L, dot, (#), (#>))

import           Numeric.SGD.ParamSet (ParamSet)

import           Graph
import           Net.Util hiding (scale)
import           Net.New
import           Net.Pair
import           Net.Graph.Arc
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))

-- import           Debug.Trace (trace)


----------------------------------------------
-- Components
----------------------------------------------


-- | Biaffinity component, responsible for scoring different labelling
-- configurations which can be assigned to a given dependency arc
class Backprop comp => BiComp dim comp where
  runBiComp 
    :: (Reifies s W)
    => Graph (BVar s (R dim)) ()
    -> Arc
    -> BVar s comp
    -> BVar s (Vec8 Pot)

instance (BiComp dim comp1, BiComp dim comp2)
  => BiComp dim (comp1 :& comp2) where
  runBiComp graph arc (comp1 :&& comp2) =
    runBiComp graph arc comp1 + runBiComp graph arc comp2


----------------------------------------------
----------------------------------------------


-- | No biaffinity (score fixed to 0.0)
data NoBi = NoBi
  deriving (Show, Generic, Binary, NFData, ParamSet)

instance Backprop NoBi

instance New a b NoBi where
  new _ _ = pure NoBi

instance BiComp dim NoBi where
  runBiComp _ _ _ = BP.auto (Vec 0.0)


----------------------------------------------
----------------------------------------------


-- | Global bias
newtype Bias = Bias
  { _biasVal :: Vec8 Pot
  } deriving (Show, Generic)
    deriving newtype (Binary, NFData, ParamSet)
instance Backprop Bias
makeLenses ''Bias
instance New a b Bias where
  new xs ys = Bias <$> new xs ys
instance BiComp dim Bias where
  runBiComp _ _ bias = bias ^^. biasVal


----------------------------------------------
----------------------------------------------


-- | Generic word affinity component; allows to make the potentials of the
-- different arc labelling configurations depend on a particular node (head,
-- dependent, grandparent, etc.).
data WordAff d h = WordAff
  { _wordAffN :: FFN d h 8
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat dim, KnownNat h) => Backprop (WordAff dim h)
makeLenses ''WordAff

instance (KnownNat dim, KnownNat h) => New a b (WordAff dim h) where
  new xs ys = WordAff <$> new xs ys


-- | Affinity of the given node
nodeAff :: (KnownNat d, KnownNat h, Reifies s W)
        => Graph (BVar s (R d)) ()
        -> G.Vertex
        -> BVar s (WordAff d h)
        -> BVar s (Vec8 Pot)
nodeAff graph v aff =
  let wordRepr = (nodeLabelMap graph M.!)
      hv = wordRepr v
   in BP.coerceVar $ FFN.run (aff ^^. wordAffN) hv


-- | Affinity of the head
newtype HeadAff d h = HeadAff { _unHeadAff :: WordAff d h }
  deriving (Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''HeadAff
instance (KnownNat dim, KnownNat h) => New a b (HeadAff dim h) where
  new xs ys = HeadAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim (HeadAff dim h) where
  runBiComp graph (_, w) aff = nodeAff graph w (aff ^^. unHeadAff)


-- | Affinity of the dependent
newtype DepAff d h = DepAff { _unDepAff :: WordAff d h }
  deriving (Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''DepAff
instance (KnownNat dim, KnownNat h) => New a b (DepAff dim h) where
  new xs ys = DepAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim (DepAff dim h) where
  runBiComp graph (v, _) aff = nodeAff graph v (aff ^^. unDepAff)


-- | Symmetric affinity (head and dependent both used in the same way)
newtype SymAff d h = SymAff { _unSymAff :: WordAff d h }
  deriving (Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''SymAff
instance (KnownNat dim, KnownNat h) => New a b (SymAff dim h) where
  new xs ys = SymAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim (SymAff dim h) where
  runBiComp graph (v, w) aff
    = nodeAff graph v (aff ^^. unSymAff)
    + nodeAff graph w (aff ^^. unSymAff)


----------------------------------------------
----------------------------------------------


-- | Biaffinity component
data BiAff d h = BiAff
  { _biAffN :: FFN (d Nats.+ d) h 8
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''BiAff

instance (KnownNat dim, KnownNat h) => New a b (BiAff dim h) where
  new xs ys = BiAff <$> new xs ys

instance (KnownNat dim, KnownNat h) => BiComp dim (BiAff dim h) where
  runBiComp graph (v, w) bi =
    let wordRepr = (nodeLabelMap graph M.!)
        hv = wordRepr v
        hw = wordRepr w
     in BP.coerceVar $ FFN.run (bi ^^. biAffN) (hv # hw)


----------------------------------------------
----------------------------------------------


-- | A version of `BiAff` where the decision about the labling of the arc
-- and its end nodes is taken partially independently.
--
--   * @d@ -- word embedding dimension
--   * @l@ -- label (POS, DEP) embedding dimension
--   * @h@ -- hidden dimension
--
data BiAffMix d h = BiAffMix
  { _biAffMixL1 :: {-# UNPACK #-} !(L h (d Nats.+ d))
    -- ^ First layer transforming the input vectors into the hidden
    -- representation
  , _biAffMixB1 :: {-# UNPACK #-} !(R h)
    -- ^ The bias corresponding to the first layer
  , _biAffMixL2_8 :: {-# UNPACK #-} !(L 8 h)
    -- ^ Second layer with joint potential attribution
  , _biAffMixB2_8 :: {-# UNPACK #-} !(R 8)
    -- ^ The bias corresponding to the second (joint) layer
  , _biAffMixL2_3 :: {-# UNPACK #-} !(L 3 h)
    -- ^ Second layer with independent potential attribution
  , _biAffMixB2_3 :: {-# UNPACK #-} !(R 3)
    -- ^ The bias corresponding to the second (independent) layer
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''BiAffMix

instance (KnownNat d, KnownNat h)
  => New a b (BiAffMix d h) where
  new xs ys = BiAffMix
    <$> new xs ys
    <*> new xs ys
    <*> new xs ys
    <*> new xs ys
    <*> new xs ys
    <*> new xs ys

instance (KnownNat d, KnownNat h)
  => BiComp d (BiAffMix d h) where

  runBiComp graph (v, w) bi =
    
    -- helper functions
    let nodeMap = nodeLabelMap graph
        emb = (nodeMap M.!)
        hv = emb v
        hw = emb w

        -- input layer
        x = hv # hw
        -- second layer
        -- y = LBP.vmap' leakyRelu $ (bi ^^. biAffMixL1) #> x + (bi ^^. biAffMixB1)
        y = leakyRelu $ (bi ^^. biAffMixL1) #> x + (bi ^^. biAffMixB1)
        -- joint output
        z8 = BP.coerceVar $
          (bi ^^. biAffMixL2_8) #> y + (bi ^^. biAffMixB2_8)
        -- independent output
        z3 = BP.coerceVar $
          (bi ^^. biAffMixL2_3) #> y + (bi ^^. biAffMixB2_3)

--      -- combine the two outputs to get the result
--      -- (with the first element zeroed out)
--      in  BP.coerceVar (BP.auto mask0) * inject z3 z8

     -- combine the two outputs to get the result
     -- (all elements with arc label=0 are zeroed out)
     in  BP.coerceVar (BP.auto mask1) * inject z3 z8
