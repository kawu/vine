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


module Net.Graph.UniComp
  ( UniComp (..)
  , Bias (..)
  , UniAff (..)
  , PairAffLeft (..)
  , PairAffRight (..)
  , NoUni
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

import qualified Test.SmallCheck.Series as SC

import           Numeric.SGD.ParamSet (ParamSet)

import           Graph
import           Net.Util hiding (scale)
import           Net.New
import           Net.Pair
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))

import           Debug.Trace (trace)


----------------------------------------------
-- Utils
----------------------------------------------


onLeft :: G.Vertex -> Graph a b -> Maybe G.Vertex
onLeft v g = fst <$> M.lookupLT v (nodeLabelMap g)


onRight :: G.Vertex -> Graph a b -> Maybe G.Vertex
onRight v g = fst <$> M.lookupGT v (nodeLabelMap g)


----------------------------------------------
-- Components
----------------------------------------------


-- | Affinity component
class Backprop bp => UniComp d bp where
  -- | Calculate the potential of the given node within the context of the
  -- given graph
  runUniComp
    :: (Reifies s W)
    => Graph (BVar s (R d)) ()
    -> G.Vertex
    -> BVar s bp
    -> BVar s Double

instance (UniComp d bp1, UniComp d bp2)
  => UniComp d (bp1 :& bp2) where
  runUniComp graph arc (comp1 :&& comp2) =
    runUniComp graph arc comp1 + runUniComp graph arc comp2


----------------------------------------------
----------------------------------------------


-- | Global bias
newtype Bias = Bias
  { _biasVal :: Double
  } deriving (Show, Generic)
    deriving newtype (Binary, NFData, ParamSet)
instance Backprop Bias
makeLenses ''Bias
instance New a b Bias where
  new xs ys = Bias <$> new xs ys
instance UniComp d Bias where
  runUniComp _ _ bias = bias ^^. biasVal


----------------------------------------------
----------------------------------------------


-- | Word affinity component
data UniAff d h = UniAff
  { _uniAffN :: FFN d h 1
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat d, KnownNat h) => Backprop (UniAff d h)
makeLenses ''UniAff

instance (KnownNat d, KnownNat h) => New a b (UniAff d h) where
  new xs ys = UniAff <$> new xs ys

instance (KnownNat dim, KnownNat h) => UniComp dim (UniAff dim h) where
  runUniComp graph v uni =
    let wordRepr = (nodeLabelMap graph M.!)
        hv = wordRepr v
        vec = LBP.extractV $ FFN.run (uni ^^. uniAffN) hv
     in vec `at` 0


-- TODO: repeated several times in different modules!
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
----------------------------------------------


-- | Word affinity component
data PairAffLeft d h = PairAffLeft
  { _pairAffLeftN :: FFN (d Nats.+ d) h 1
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat d, KnownNat h) => Backprop (PairAffLeft d h)
makeLenses ''PairAffLeft

instance (KnownNat d, KnownNat h) => New a b (PairAffLeft d h) where
  new xs ys = PairAffLeft <$> new xs ys

instance (KnownNat dim, KnownNat h) => UniComp dim (PairAffLeft dim h) where
  runUniComp graph v pair =
    maybe 0 id $ doIt v <$> onRight v graph
    where
      doIt v w =
        let wordRepr = (nodeLabelMap graph M.!)
            hv = wordRepr v
            hw = wordRepr w
            vec = LBP.extractV $ FFN.run (pair ^^. pairAffLeftN) (hv # hw)
         in vec `at` 0


----------------------------------------------
----------------------------------------------


-- | Word affinity component
data PairAffRight d h = PairAffRight
  { _pairAffRightN :: FFN (d Nats.+ d) h 1
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat d, KnownNat h) => Backprop (PairAffRight d h)
makeLenses ''PairAffRight

instance (KnownNat d, KnownNat h) => New a b (PairAffRight d h) where
  new xs ys = PairAffRight <$> new xs ys

instance (KnownNat dim, KnownNat h) => UniComp dim (PairAffRight dim h) where
  runUniComp graph v pair =
    maybe 0 id $ doIt v <$> onRight v graph
    where
      doIt v w =
        let wordRepr = (nodeLabelMap graph M.!)
            hv = wordRepr v
            hw = wordRepr w
            vec = LBP.extractV $ FFN.run (pair ^^. pairAffRightN) (hv # hw)
         in vec `at` 0


----------------------------------------------
----------------------------------------------


-- | Word affinity component
data NoUni = NoUni
  deriving (Generic, Binary, NFData, ParamSet)

instance Backprop NoUni
makeLenses ''NoUni

instance New a b NoUni where
  new _ _ = pure NoUni

instance (KnownNat dim) => UniComp dim NoUni where
  runUniComp _ _ _ = 0.0
