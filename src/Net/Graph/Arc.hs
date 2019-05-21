{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveFoldable #-}


-- | The module provides different representations of complex, arc-related
-- labels, together with the corresponding probabilities and potentials.


module Net.Graph.Arc
  ( 
  -- * Types
    Pot
  , Prob
  , Vec(..)
  , Vec8
  , Out(..)
  
  -- * Functions
  , enumerate
  , mask
  , squash
  , explicate
  , obfuscate
  , encode
  , decode

  -- * Internal
  , inject
  , mask0
  , mask1
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)

import           Control.DeepSeq (NFData)

import           Data.Binary (Binary)
import qualified Data.Map.Strict as M

import qualified Test.SmallCheck.Series as SC

import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, Reifies, W, BVar) --, (^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop (R, dot) --, L, (#), (#>))

import           Numeric.SGD.ParamSet (ParamSet)

import           Net.Util (toList, at)
import           Net.New


----------------------------------------------
-- Vector 8
----------------------------------------------


-- | Potential/probability annotation
data Pot
data Prob


-- | A static-length vector with potentials (`Pot`) or probabilities (`Prob`).
newtype Vec n p = Vec { unVec :: R n }
  deriving (Show, Generic)
  deriving newtype (Binary, NFData, ParamSet, Num, Backprop)

instance (KnownNat n) => New a b (Vec n p) where
  new xs ys = Vec <$> new xs ys


-- | Type synonym for @Vec 8 p@.
type Vec8 p = Vec 8 p


----------------------------------------------
-- Out
--
-- Note that @Out Double@ has roughly the same semantics as @Vec 3@, i.e, both
-- represent a vector of 3 values.  The reasons to have two different types
-- are:
--
-- * We can have a `BVar` vector, but it is not possible to have a vector of
--   `BVar`'s.  This is perfectly possible with the `Out` type.  It is thus
--   sometimes more convenient to use `Out`, because it allows to easily
--   select the desired component value.
-- * `Out` is also "more" polymorphic.  You can, for instance, create an `Out`
--    of `Bool`ean values, which can represent a particular labeling of an arc
--    and the connected nodes.
-- * With @Vec 3@, on the other hand, we can easily specify whether the values inside
--   represent probabilities (`Prob`) or potentials (`Pot`).  This is currently not
--   possible with `Out` (although `Out` could be probably improved in this respect
--   if needed).
--
----------------------------------------------


-- | Output structure in which a value of type @a@ is assigned to an arc and
-- the two nodes it connects.
data Out a = Out
  { arcVal :: a
    -- ^ Value assigned to the arc
  , hedVal :: a
    -- ^ Value assigned to the head
  , depVal :: a
    -- ^ Value assigned to the dependent
  } deriving (Generic, Show, Eq, Ord, Functor, Foldable)

-- Allows to use SmallCheck to test (decode . encode) == id.
instance (SC.Serial m a) => SC.Serial m (Out a)


-- | Enumerate the possible arc/node labelings in order consistent with the
-- encoding/decoding format.  Consistent in the sense that
--
--   @zip enumerate (toList $ unVec vec)@
--
-- provides a list in which to each `Out` value the corresponding value from
-- the vector @vec@ is assigned.
--
enumerate :: [Out Bool]
enumerate = do
  b1 <- [False, True]
  b2 <- [False, True]
  b3 <- [False, True]
  return $ Out b1 b2 b3


-- | A mask vector which allows to easily obtain (with dot product) the
-- potential of a given `Out` labeling.  The following property should be
-- satisfied:
--
--   @mask (enumerate !! i) ! j == (i == j)@
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


----------------------------------------------
-- Conversion @Vec8@ <-> @Out@
----------------------------------------------


-- | Determine the values assigned to different labellings of the given arc and
-- nodes.
explicate :: Vec8 p -> M.Map (Out Bool) Double
explicate = M.fromList . zip enumerate . toList . unVec


-- | The inverse of `explicate`.
obfuscate :: M.Map (Out Bool) Double -> Vec8 p
obfuscate = Vec . LA.vector . M.elems


-- | Decode the output structure from the given probability vector.  This
-- function is potentially lossy in the sense that @Vec8 Prob@ encodes a joint
-- distibution and the resulting @Out Double@ encodes three distributions
-- assumed to be independent.
decode :: Vec8 Prob -> Out Double
decode = BP.evalBP $ BP.collectVar . squash
--   case decode' (unVec vec) of
--     [arcP, hedP, depP] -> Out
--       { arcVal = arcP
--       , hedVal = hedP
--       , depVal = depP
--       }
--     xs -> error $
--       "Graph.decode: unsupported list length (" ++
--        show xs ++ ")"


-- -- | Decode the list of probabilities/potentials from the given vector.
-- decode'
--   :: (KnownNat n)
--   => R n
--   -> [Double]
-- decode' =
--   go . toList
--   where
--     go []  = []
--     go [_] = []
--     go xs =
--       let (left, right) = rightInTwo xs
--           -- p0 = sum left
--           p1 = sum right
--        in p1 : go (map (uncurry (+)) (zip left right))


-- | The `squash` function is a backpropagation-enabled version of `decode`.
-- The result is a structure with three probability values:
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


-- | Encode the output structure *with probabilities* as a vector.
--
-- TODO: Out Double -> e.g. Out (Real Prob) or Out (Float Prob)
--
encode :: Out Double -> Vec8 Prob
encode Out{..} = (Vec . encode') [arcVal, hedVal, depVal]


-- | Encode a list of probabilities of length @n@ as a vector of length @2^n@.
encode'
  :: (KnownNat n)
  => [Double]
  -> R n
encode' =
  LA.vector . go
  where
    go (p:ps)
      = map (uncurry (*))
      $ cartesian [1-p, p] (go ps)
    go [] = [1]


-- | Cartesian product of two lists
cartesian :: [a] -> [b] -> [(a, b)]
cartesian xs ys = do
  x <- xs
  y <- ys
  return (x, y)


----------------------------------------------
-- Conversion @Vec3@ -> @Vec8@
----------------------------------------------


-- | V3 -> V8 expansion
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
-- TODO: What was the purpose of the NOINLINE pragmas?


-- | Combine the independent with the joint potential vector (a type-safe
-- wrapper over inject').
inject
  :: (Reifies s W)
  => BVar s (Vec 3 Pot)
  -> BVar s (Vec 8 Pot)
  -> BVar s (Vec 8 Pot)
inject v3 v8 = expand v3 + v8
{-# INLINE inject #-}
