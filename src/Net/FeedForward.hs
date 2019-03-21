{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
-- {-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

-- To derive Binary
{-# LANGUAGE DeriveAnyClass #-}


{- Feed-forward network -}


module Net.FeedForward
  ( FFN (..)
  -- , new
  , run
  -- , substract
  ) where

import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)

import           Control.DeepSeq (NFData)

import           Lens.Micro.TH (makeLenses)

import           Data.Binary (Binary)

import qualified Numeric.Backprop as BP
import           Numeric.Backprop ((^^.))
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W)
import           Numeric.LinearAlgebra.Static.Backprop ((#>))

import           Numeric.SGD.ParamSet (ParamSet)

import           Net.Basic
import           Net.ArcGraph.Graph (New(..))


----------------------------------------------
-- Network layer
----------------------------------------------


-- data Layer i o = Layer
--   { _nWeights :: {-# UNPACK #-} !(L o i)
--   , _nBias    :: {-# UNPACK #-} !(R o)
--   }
--   deriving (Show, Generic, Binary)
-- 
-- instance (KnownNat i, KnownNat o) => BP.Backprop (Layer i o)
-- makeLenses ''Layer
-- instance (KnownNat i, KnownNat o) => ParamSet (Layer i o)
-- instance (KnownNat i, KnownNat o) => NFData (Layer i o)
-- instance (KnownNat i, KnownNat o) => New a b (Layer i o) where
--   new xs ys = Layer <$> new xs ys <*> new xs ys
-- 
-- runLayer
--   :: (KnownNat i, KnownNat o, Reifies s W)
--   => BVar s (Layer i o)
--   -> BVar s (R i) 
--   -> BVar s (R o)
-- runLayer l x = (l ^^. nWeights) #> x + (l ^^. nBias)
-- {-# INLINE runLayer #-}


----------------------------------------------
-- Feed-forward net
----------------------------------------------


data FFN idim hdim odim = FFN
  { _nWeights1 :: {-# UNPACK #-} !(L hdim idim)
  , _nBias1    :: {-# UNPACK #-} !(R hdim)
  , _nWeights2 :: {-# UNPACK #-} !(L odim hdim)
  , _nBias2    :: {-# UNPACK #-} !(R odim)
  }
  deriving (Show, Generic, Binary)

instance (KnownNat idim, KnownNat hdim, KnownNat odim) 
  => BP.Backprop (FFN idim hdim odim)
makeLenses ''FFN
instance (KnownNat i, KnownNat h, KnownNat o)
  => ParamSet (FFN i h o)
instance (KnownNat i, KnownNat h, KnownNat o)
  => NFData (FFN i h o)
instance (KnownNat i, KnownNat h, KnownNat o) => New a b (FFN i h o) where
  new xs ys = FFN
    <$> new xs ys
    <*> new xs ys
    <*> new xs ys
    <*> new xs ys

run
  :: (KnownNat idim, KnownNat hdim, KnownNat odim, Reifies s W)
  => BVar s (FFN idim hdim odim)
  -> BVar s (R idim)
  -> BVar s (R odim)
run net x = z
  where
    -- run first layer
    y = leakyRelu $ (net ^^. nWeights1) #> x + (net ^^. nBias1)
    -- run second layer
    z = (net ^^. nWeights2) #> y + (net ^^. nBias2)


-- ----------------------------------------------
-- -- Feed-forward net: version with layers
-- ----------------------------------------------
-- 
-- 
-- data FFN idim hdim odim = FFN
--   { _nLayer1 :: {-# UNPACK #-} !(Layer idim hdim)
--   , _nLayer2 :: {-# UNPACK #-} !(Layer hdim odim)
--   }
--   deriving (Show, Generic, Binary)
-- 
-- instance (KnownNat idim, KnownNat hdim, KnownNat odim) 
--   => BP.Backprop (FFN idim hdim odim)
-- makeLenses ''FFN
-- instance (KnownNat i, KnownNat h, KnownNat o)
--   => ParamSet (FFN i h o)
-- instance (KnownNat i, KnownNat h, KnownNat o)
--   => NFData (FFN i h o)
-- instance (KnownNat i, KnownNat h, KnownNat o) => New a b (FFN i h o) where
--   new xs ys = FFN <$> new xs ys <*> new xs ys
-- 
-- 
-- run
--   :: (KnownNat idim, KnownNat hdim, KnownNat odim, Reifies s W)
--   => BVar s (FFN idim hdim odim)
--   -> BVar s (R idim)
--   -> BVar s (R odim)
-- run net x = z
--   where
--     -- run first layer
--     y = leakyRelu $ runLayer (net ^^. nLayer1) x
--     -- run second layer
--     z = runLayer (net ^^. nLayer2) y
