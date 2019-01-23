{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}


module Decoder
  ( Decoder(..)
  , new
  , run
  , substract
  ) where


import           Prelude hiding (words)

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

import           Basic
import qualified FeedForward as FFN
import           FeedForward (FFN(..))
import qualified GradientDescent as GD


----------------------------------------------
-- Decoder
----------------------------------------------


-- | Decoder RNN
data Decoder h gh i bh = Decoder
  { _ffG :: FFN
      h  -- RNN's hidden state
      gh  -- ffG's internal hidden state
      i  -- ffG's output size (size of the vocabulary, including EOS)
  , _ffB :: FFN
      -- 10 -- ffB takes on input the current word + the previous RNN's hidden state
      (i Nats.+ h) -- ffB takes on input the current word + the previous RNN's hidden state
      bh  -- ffB's internal hidden state
      h   -- ffB's output size (the next RNN's hidden state)
--   , _h0  :: R h
--     -- ^ The initial RNN's hidden state
  , _eos :: R i
    -- ^ Vector representation of EOS (WARNING: `eos` has to be constant!)
  } deriving (Generic)

instance
  ( KnownNat h, KnownNat gh
  , KnownNat i, KnownNat bh, KnownNat (i Nats.+ h)) 
  => BP.Backprop (Decoder h gh i bh)

makeLenses ''Decoder


-- | Create a new, random Decoder
new
  :: ( KnownNat h, KnownNat gh
     , KnownNat i, KnownNat bh, KnownNat (i Nats.+ h) )
  => Int -- h
  -> Int -- gh
  -> Int -- i
  -> Int -- bh
  -> R i -- EOS
  -> IO (Decoder h gh i bh)
new h gh i bh eos =
  Decoder <$> FFN.new h gh i <*> FFN.new (i+h) bh h <*> pure eos


run
  :: ( KnownNat h, KnownNat gh, KnownNat i
     , KnownNat bh, KnownNat (i Nats.+ h)
     , Reifies s W )
  => BVar s (Decoder h gh i bh)
--   -> BVar s (R i)
--     -- ^ EOS representation
  -> BVar s (R h)
    -- ^ Input hidden value (coming from the encoder)
  -> [BVar s (R i)]
    -- ^ Sentence (sequence of vector representations)
  -> BVar s Double
    -- ^ Probability of the sentence (log-domain!)
run net h0 =
  go h0 (log 1.0)
  where
    -- run the calculation, given the previous hidden state
    -- and the list of words to generate
    go hPrev prob (wordVect : ws) =
      let
        -- determine the probability vector
        probVect = softmax $ FFN.run (net ^^. ffG) hPrev
        -- determine the actual probability of the current word
        newProb = log $ probVect `LBP.dot` wordVect
        -- determine the next hidden state
        hNext = FFN.run (net ^^. ffB) (wordVect # hPrev)
      in
        go hNext (prob + newProb) ws
    go hPrev prob [] =
      let
        -- determine the probability vector
        probVect = softmax $ FFN.run (net ^^. ffG) hPrev
        -- determine the actual probability of EOS
        newProb = log $ probVect `LBP.dot` (net ^^. eos)
      in
        prob + newProb


-- | Substract the second network from the first one.
substract
  :: ( KnownNat h, KnownNat gh, KnownNat i
     , KnownNat bh, KnownNat (i Nats.+ h) )
  => Decoder h gh i bh
  -> Decoder h gh i bh
  -> Double 
  -> Decoder h gh i bh
substract x y coef = Decoder
  { _ffG = FFN.substract (_ffG x) (_ffG y) coef
  , _ffB = FFN.substract (_ffB x) (_ffB y) coef
  -- , _h0 = _h0 x - scale (_h0 y)
  , _eos = _eos x -- EOS should not change; this is a bit risky!
  }
--   where
--     scale x
--       = fromJust
--       . LA.create
--       . LAD.scale coef
--       $ LA.unwrap x
