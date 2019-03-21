{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE TypeFamilies #-}


module Net.Basic
  ( logistic
  , sigma
  , reluSmooth
  , leakyRelu
  , relu
  , softmax
  -- * Utils
  -- , elemWiseMult
  , randomList
  , matrix
  , vector
  , scale
  , vec1
  , elem0
  , elem1
  , elem2
  , toList
  ) where


import           GHC.TypeNats (KnownNat)
import qualified GHC.TypeNats as Nats

import           System.Random (randomRIO)

import           Data.Maybe (fromJust)
import qualified Data.Vector.Storable.Sized as SVS
import qualified Numeric.Backprop as BP
import           Numeric.Backprop (BVar, Reifies, W)
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop (R, L)
import qualified Numeric.LinearAlgebra as LAD
import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra.Static.Vector as LA

-- -- To make GHC automatically infer that, e.g., `KnownNat d => KnownNat (d + d)`
-- {-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}


------------------------------------
-- Activation functions
------------------------------------


logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))


-- TODO: are you sure?
sigma :: Floating a => a -> a
sigma = logistic


reluSmooth :: Floating a => a -> a
reluSmooth x = log(1 + exp(x))


relu :: Floating a => a -> a
-- relu x = max 0 x
relu x = (x + abs x) / 2


leakyRelu :: Floating a => a -> a
leakyRelu x
  = relu x
  + 0.01 * relu (-x)
-- leakyRelu :: (Ord a, Floating a) => a -> a
-- leakyRelu x
--   | x < 0 = 0.01*x
--   | otherwise = x


-- | Apply the softmax layer to a vector.
softmax
  :: (KnownNat n, Reifies s W)
  => BVar s (R n)
  -> BVar s (R n)
softmax x0 =
  LBP.vmap (/norm) x
  where
    x = LBP.vmap' exp x0
    norm = LBP.norm_1V x


------------------------------------
-- Utils
------------------------------------



-- -- | Element-wise multiplication
-- --
-- -- TODO: Make sure this is correct!
-- --
-- elemWiseMult
--   :: (KnownNat n, Reifies s W)
--   => BVar s (R n)
--   -> BVar s (R n)
--   -> BVar s (R n)
-- elemWiseMult x y = x * y


-- | A random list of values between 0 and 1
randomList :: Int -> IO [Double]
randomList 0 = return []
randomList n = do
  -- NOTE:  (-0.1, 0.1) worked well, checking smaller values
  -- NOTE:  (-0.01, 0.01) seems to work fine as well
  r  <- randomRIO (-0.01, 0.01)
  rs <- randomList (n-1)
  return (r:rs) 


-- | Create a random matrix
matrix
  :: (KnownNat n, KnownNat m)
  => Int -> Int -> IO (L m n)
matrix n m = do
  list <- randomList (n*m)
  return $ LA.matrix list


-- | Create a random vector
vector :: (KnownNat n) => Int -> IO (R n)
vector k = do
  list <- randomList k
  return $ LA.vector list


-- | Scale the given vector/matrix
scale
  :: (LAD.Linear t d, LA.Sized t c d, LA.Sized t s d) 
  => t -> s -> c
scale coef x = fromJust . LA.create . LAD.scale coef $ LA.unwrap x
{-# INLINE scale #-}


-- | Create a singleton vector (an overkill, but this
-- should be provided in the upstream libraries)
vec1 :: Reifies s W => BVar s Double -> BVar s (R 1)
vec1 =
  BP.isoVar
    (LA.vector . (:[]))
    (\(LA.rVec->v) -> (SVS.index v 0))
{-# INLINE vec1 #-}


-- | Extract the @0@ (first!) element of the given vector.
elem0
  :: (Reifies s W, KnownNat n, 1 Nats.<= n)
  => BVar s (R n) -> BVar s Double
elem0 = fst . LBP.headTail
{-# INLINE elem0 #-}


-- | Extract the @1@-th (second!) element of the given vector.
elem1
  :: ( Reifies s W
     , KnownNat (n Nats.- 1), KnownNat n
     , (1 Nats.<=? (n Nats.- 1)) ~ 'True
     , (1 Nats.<=? n) ~ 'True 
     )
  => BVar s (R n) -> BVar s Double
elem1 = fst . LBP.headTail . snd . LBP.headTail
{-# INLINE elem1 #-}


-- | Extract the @2@ (third!) element of the given vector.
elem2
  :: ( Reifies s W
     , KnownNat (n Nats.- 1), KnownNat n
     , KnownNat ((n Nats.- 1) Nats.- 1)
     , (1 Nats.<=? ((n Nats.- 1) Nats.- 1)) ~ 'True
     , (1 Nats.<=? (n Nats.- 1)) ~ 'True
     , (1 Nats.<=? n) ~ 'True
     )
  => BVar s (R n) -> BVar s Double
elem2 = fst . LBP.headTail . snd . LBP.headTail . snd . LBP.headTail
{-# INLINE elem2 #-}


-- | Convert the given vector to a list
toList :: (KnownNat n) => R n -> [Double]
toList = LAD.toList . LA.unwrap
