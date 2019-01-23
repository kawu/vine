{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}


module Basic
  ( logistic
  , reluSmooth
  , relu
  , softmax
  -- * Utils
  , matrix
  , vector
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
import qualified GHC.TypeNats as Nats

import           Control.Monad (forM, forM_)

import           System.Random (randomRIO)

-- import           Control.Lens.TH (makeLenses)
import           Lens.Micro.TH (makeLenses)
import           Lens.Micro ((^.))

import           Data.Maybe (fromJust)
import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#))
import qualified Numeric.LinearAlgebra as LAD
import qualified Numeric.LinearAlgebra.Static as LA
import           Numeric.LinearAlgebra.Static.Backprop ((#>))
import qualified Debug.SimpleReflect as Refl


logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))


reluSmooth :: Floating a => a -> a
reluSmooth x = log(1 + exp(x))


relu :: Floating a => a -> a
-- relu x = max 0 x
relu x = (x + abs x) / 2


-- | Apply the softmax layer to a vector
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


-- | A random list of values between 0 and 1
randomList :: Int -> IO [Double]
randomList 0 = return []
randomList n = do
  r  <- randomRIO (0, 1)
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

