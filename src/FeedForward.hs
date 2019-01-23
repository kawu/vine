{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}


{- Feed-forward network -}


module FeedForward
  ( FFN (..)
  , run
  , substract
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

import Basic


----------------------------------------------
-- Feed-forward net
----------------------------------------------


data FFN idim hdim odim = FFN
  { _nWeights1 :: L hdim idim
  , _nBias1    :: R hdim
  , _nWeights2 :: L odim hdim
  , _nBias2    :: R odim
  }
  deriving (Show, Generic)

instance (KnownNat idim, KnownNat hdim, KnownNat odim) 
  => BP.Backprop (FFN idim hdim odim)

makeLenses ''FFN


run
  :: (KnownNat idim, KnownNat hdim, KnownNat odim, Reifies s W)
  => BVar s (FFN idim hdim odim)
  -> BVar s (R idim)
  -> BVar s (R odim)
run net x = z
  where
    -- run first layer
    y = logistic $ (net ^^. nWeights1) #> x + (net ^^. nBias1)
    -- run second layer
    z = relu $ (net ^^. nWeights2) #> y + (net ^^. nBias2)
    -- z = (net ^^. nWeights2) #> y + (net ^^. nBias2)


-- | Substract the second network from the first one.
substract 
  :: (KnownNat idim, KnownNat hdim, KnownNat odim)
  => FFN idim hdim odim
  -> FFN idim hdim odim
  -> Double 
  -> FFN idim hdim odim
substract x y coef = FFN
  { _nWeights1 = _nWeights1 x - scale (_nWeights1 y)
  , _nBias1 = _nBias1 x - scale (_nBias1 y)
  , _nWeights2 = _nWeights2 x - scale (_nWeights2 y)
  , _nBias2 = _nBias2 x - scale (_nBias2 y)
  }
  where
    scale x
      = fromJust
      . LA.create
      . LAD.scale coef
      $ LA.unwrap x
