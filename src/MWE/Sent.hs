{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}


-- | Top-level sentence representation


module MWE.Sent
  ( Sent (..)
  ) where


import           GHC.Generics (Generic)
import           Data.Binary (Binary)
import           Numeric.LinearAlgebra.Static.Backprop (R)

import qualified Format.Cupt as Cupt


-- | Input sentence
data Sent d = Sent
  { cuptSent :: Cupt.Sent
    -- ^ The .cupt sentence
  , wordEmbs :: [R d]
    -- ^ The corresponding word embeddings
  } deriving (Show, Generic, Binary)
