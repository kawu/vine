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
  ( Pot
  , Prob
  , Vec(..)
  , Vec8
  , softmaxVec
  
  , BiComp (..)
  , Bias (..)

  , BiAff (..)
  , BiAffMix (..)

  , NoBi (..)

  -- * Vec3 <-> Vec8 conversion
  , Out(..)
  , enumerate
  , mask
  , squash
  -- , stretch
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
-- Various Utils
----------------------------------------------


-- | Remove duplicates.
nub :: (Ord a) => [a] -> [a]
nub = S.toList . S.fromList


----------------------------------------------
-- Vec8 <-> Vec3 conversion
----------------------------------------------


-- -- | V8 -> V3 squashing
-- --
-- -- The `squash` function is a backpropagation-enabled version of @decode@ from
-- -- `Net.Graph`.  The result is a vector of three probability values:
-- --
-- --   * Probability of the arc being a MWE
-- --   * Probability of the head being a MWE
-- --   * Probability of the dependent being a MWE
-- --
-- squash
--   :: (Reifies s W)
--   => BVar s (Vec 8 Prob)
--   -> BVar s (R 3)
-- squash v8 = undefined


-- | Output structure in which a value of type @a@ is assigned to an arc and
-- the two nodes it connects.
data Out a = Out
  { arcVal :: a
    -- ^ Value assigned to the arc
  , hedVal :: a
    -- ^ Value assigned to the head
  , depVal :: a
    -- ^ Value assigned to the dependent
  } deriving (Generic, Show, Eq, Ord, Functor, Foldable) --, Traversable)

-- Allows to use SmallCheck to test (decode . encode) == id.
instance (SC.Serial m a) => SC.Serial m (Out a)


-- | Enumerate the possible arc/node labelings in order consistent with the
-- encoding/decoding format.
enumerate :: [Out Bool]
enumerate = do
  b1 <- [False, True]
  b2 <- [False, True]
  b3 <- [False, True]
  return $ Out b1 b2 b3


-- | A mask vector which allows to easily obtain (with dot product) the
-- potential of a given `Out` labeling.
--
-- TODO: the mask could be perhaps calculated using bit-level operattions?
--
mask :: Out Bool -> R 8
mask (Out False False False) = vec18
mask (Out False False True)  = vec28
mask (Out False True  False) = vec38
mask (Out False True  True)  = vec48
mask (Out True  False False) = vec58
mask (Out True  False True)  = vec68
mask (Out True  True  False) = vec78
mask (Out True  True  True)  = vec88


-- | Hard-coded masks
vec18, vec28, vec38, vec48, vec58, vec68, vec78, vec88 :: R 8
vec18 = LA.vector [1, 0, 0, 0, 0, 0, 0, 0]
vec28 = LA.vector [0, 1, 0, 0, 0, 0, 0, 0]
vec38 = LA.vector [0, 0, 1, 0, 0, 0, 0, 0]
vec48 = LA.vector [0, 0, 0, 1, 0, 0, 0, 0]
vec58 = LA.vector [0, 0, 0, 0, 1, 0, 0, 0]
vec68 = LA.vector [0, 0, 0, 0, 0, 1, 0, 0]
vec78 = LA.vector [0, 0, 0, 0, 0, 0, 1, 0]
vec88 = LA.vector [0, 0, 0, 0, 0, 0, 0, 1]


-- -- | Cross entropy between the true and the artificial distributions
-- crossEntropy
--   :: forall s. (Reifies s W)
--   => BVar s (Vec 8 Prob)
--     -- ^ Target ,,true'' probability distribution
--   -> BVar s (Vec 8 Prob)
--     -- ^ Output ,,artificial'' distribution
--   -> BVar s Double
-- crossEntropy p0 q0 =
--   undefined


-- | The `squash` function is a backpropagation-enabled version of @decode@
-- from `Net.Graph`.  The result is a vector of three probability values:
--
--   * Probability of the arc being a MWE
--   * Probability of the head being a MWE
--   * Probability of the dependent being a MWE
--
squash :: forall s. (Reifies s W) => BVar s (Vec 8 Prob) -> Out (BVar s Double)
squash v8_vec = Out
  { arcVal = BP.auto mask1 `dot` v8
  , hedVal = BP.auto mask2 `dot` v8
  , depVal = BP.auto mask3 `dot` v8
  } 
  where
    v8 = BP.coerceVar v8_vec :: BVar s (R 8)


-- -- | The `stretch` function is a backpropagation-enabled version of @encode@
-- -- from `Net.Graph`.
-- stretch :: forall s. (Reifies s W) => Out (BVar s Double) -> BVar s (Vec 8 Prob)
-- stretch = undefined 


-- | V3 -> V8 expansion
--
-- TODO: Make some kind of link between `epxand` and @encode@ from
-- `Net.Graph`.
--
expand
  :: (Reifies s W)
  => BVar s (Vec 3 Pot)
  -> BVar s (Vec 8 Pot)
expand v3 = BP.coerceVar $ expand' (BP.coerceVar v3)


-- | Combine the independent with the joint potential vector (lower-level
-- function).
expand' :: (Reifies s W) => BVar s (R 3) -> BVar s (R 8)
expand' v3 
  = LBP.vmap (*x1) (BP.auto mask1)
  + LBP.vmap (*x2) (BP.auto mask2)
  + LBP.vmap (*x3) (BP.auto mask3)
  where
    v3' = LBP.extractV v3
    x1 = v3' `at` 0
    x2 = v3' `at` 1
    x3 = v3' `at` 2
{-# INLINE expand' #-}


-- | Expansion masks
mask0, mask1, mask2, mask3 :: R 8
mask0 = LA.vector [0, 1, 1, 1, 1, 1, 1, 1]
mask1 = LA.vector [0, 0, 0, 0, 1, 1, 1, 1]
mask2 = LA.vector [0, 0, 1, 1, 0, 0, 1, 1]
mask3 = LA.vector [0, 1, 0, 1, 0, 1, 0, 1]
{-# NOINLINE mask0 #-}
{-# NOINLINE mask1 #-}
{-# NOINLINE mask2 #-}
{-# NOINLINE mask3 #-}


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
-- Components
----------------------------------------------


-- | Potential/probability annotation
data Pot
data Prob


-- | Output potential is a vector of length 2^3 (the number of possible
-- combinations of values of the:
--
--   * label of the head
--   * label of the dependent
--   * label of the arc
--
-- We additionally trace the size of the vector @n@.  It's actually just for
-- convenience, to not to have to tell in other places in the code that the
-- size of `unVec` is actually @8@.
--
newtype Vec n p = Vec { unVec :: R n }
  deriving (Show, Generic)
  deriving newtype (Binary, NFData, ParamSet, Num, Backprop)

instance (KnownNat n) => New a b (Vec n p) where
  new xs ys = Vec <$> new xs ys


-- | Softmax over a vector of potentials
softmaxVec 
  :: forall n s. (KnownNat n, Reifies s W)
  => BVar s (Vec n Pot) -> BVar s (Vec n Prob)
softmaxVec
  = BP.coerceVar
  . (softmax :: Reifies s W => BVar s (R n) -> BVar s (R n))
  . BP.coerceVar


-- | Type synonym to @Vec 8 p@.
type Vec8 p = Vec 8 p


-- | Biaffinity component
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


-- | Global bias
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


-- | Word affinity component
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


newtype HeadAff d h = HeadAff { _unHeadAff :: WordAff d h }
  deriving (Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''HeadAff
instance (KnownNat dim, KnownNat h) => New a b (HeadAff dim h) where
  new xs ys = HeadAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim (HeadAff dim h) where
  runBiComp graph (_, w) aff = nodeAff graph w (aff ^^. unHeadAff)


newtype DepAff d h = DepAff { _unDepAff :: WordAff d h }
  deriving (Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''DepAff
instance (KnownNat dim, KnownNat h) => New a b (DepAff dim h) where
  new xs ys = DepAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim (DepAff dim h) where
  runBiComp graph (v, _) aff = nodeAff graph v (aff ^^. unDepAff)


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


-- | Combine the independent with the joint potential vector (a type-safe
-- wrapper over inject').
inject
  :: (Reifies s W)
  => BVar s (Vec 3 Pot)
  -> BVar s (Vec 8 Pot)
  -> BVar s (Vec 8 Pot)
inject v3 v8 = expand v3 + v8
{-# INLINE inject #-}
