{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

-- To derive Binary
{-# LANGUAGE DeriveAnyClass #-}


-- | This module provides a feed-forward network implementation.


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
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#>))

import           Numeric.SGD.ParamSet (ParamSet)

import           Net.Util
import           Net.New (New(..))


----------------------------------------------
-- Feed-forward net
----------------------------------------------


-- | Structure representing the parameters of a feed-forward neetwork.
--
--   * @i@ -- input size
--   * @i@ -- hidden layer size (number of hidden units)
--   * @o@ -- output size
--
data FFN i h o = FFN
  { _nWeights1 :: {-# UNPACK #-} !(L h i)
    -- ^ First layer matrix
  , _nBias1    :: {-# UNPACK #-} !(R h)
    -- ^ First layer bias
  , _nWeights2 :: {-# UNPACK #-} !(L o h)
    -- ^ Second layer matrix
  , _nBias2    :: {-# UNPACK #-} !(R o)
    -- ^ Second layer bias
  }
  deriving (Show, Generic, Binary)

makeLenses ''FFN

instance (KnownNat i, KnownNat h, KnownNat o) => BP.Backprop (FFN i h o)
instance (KnownNat i, KnownNat h, KnownNat o) => ParamSet (FFN i h o)
instance (KnownNat i, KnownNat h, KnownNat o) => NFData (FFN i h o)
instance (KnownNat i, KnownNat h, KnownNat o) => New a b (FFN i h o) where
  new xs ys = FFN
    <$> new xs ys
    <*> new xs ys
    <*> new xs ys
    <*> new xs ys

-- | Evaluate the network on the given input vector in the context of
-- backpropagation.
run
  :: (KnownNat i, KnownNat h, KnownNat o, Reifies s W)
  => BVar s (FFN i h o)
  -> BVar s (R i)
  -> BVar s (R o)
run net x = z
  where
    -- run first layer
    y = leakyRelu $ (net ^^. nWeights1) #> x + (net ^^. nBias1)
    -- y = LBP.vmap' leakyRelu $ (net ^^. nWeights1) #> x + (net ^^. nBias1)
    -- run second layer
    z = (net ^^. nWeights2) #> y + (net ^^. nBias2)
