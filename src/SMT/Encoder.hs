{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE StandaloneDeriving #-}


module SMT.Encoder
  ( Encoder(..)
  , new
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
                    (R, L, BVar, Reifies, W, (#), (#>))
import qualified Numeric.LinearAlgebra as LAD
import qualified Numeric.LinearAlgebra.Static as LA
import qualified Debug.SimpleReflect as Refl

import           Net.Basic
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))


----------------------------------------------
-- Encoder RNN
----------------------------------------------


-- | Encoder RNN
data Encoder i ffh h = Encoder
  { _ffn :: FFN (i Nats.+ h) ffh h
    -- ^ The underlying feed-forward network used to calculate 
    -- the subsequent hidden values
  , _h0  :: R h
    -- ^ The initial hidden state
  } deriving (Generic)

instance (KnownNat i, KnownNat ffh, KnownNat h, KnownNat (i Nats.+ h)) =>
  BP.Backprop (Encoder i ffh h)

makeLenses ''Encoder


-- | Create a new, random Encoder
new
  :: (KnownNat i, KnownNat ffh, KnownNat h, KnownNat (i Nats.+ h))
  => Int -- i
  -> Int -- ffh
  -> Int -- h
  -> IO (Encoder i ffh h)
new i ffh h =
  Encoder <$> FFN.new (i+h) ffh h <*> vector h


run
  :: (KnownNat i, KnownNat ffh, KnownNat h, KnownNat (i Nats.+ h), Reifies s W)
  => BVar s (Encoder i ffh h)
  -> [BVar s (R i)]
  -> [BVar s (R h)]
run net [] = [net ^^. h0]
run net (x:xs) = h:hs
  where
    -- run the recursive calculation (unless `null xs`)
    hs = run net xs
    -- get the last hidden state
    h' = head hs
    -- calculate the resulting hidden value
    h = FFN.run (net ^^. ffn) (x # h')


-- | Substract the second network from the first one.
substract
  :: (KnownNat i, KnownNat ffh, KnownNat h, KnownNat (i Nats.+ h))
  => Encoder i ffh h
  -> Encoder i ffh h
  -> Double
  -> Encoder i ffh h
substract x y coef = Encoder
  { _ffn = FFN.substract (_ffn x) (_ffn y) coef
  , _h0 = _h0 x - scale (_h0 y)
  }
  where
    scale x
      = fromJust
      . LA.create
      . LAD.scale coef
      $ LA.unwrap x
