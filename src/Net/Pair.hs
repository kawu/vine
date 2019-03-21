{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE PolyKinds #-}


-- | The module provides a custom pair with some useful class instances (e.g.
-- `Backprop`).


module Net.Pair
  ( (:&) (..)
  , pattern (:&&)
  ) where


import           GHC.Generics (Generic)

import           Control.DeepSeq (NFData)
import           Control.Lens (Lens)

import           Data.Binary (Binary)

import           Numeric.SGD.ParamSet (ParamSet)

import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, Reifies, W, BVar, (^^.))

import           Net.New


-- | Custom pair, with Backprop and ParamSet instances, and nice Backprop
-- pattern.
data a :& b = !a :& !b
  deriving (Show, Generic, Binary, NFData, Backprop, ParamSet)
infixr 2 :&

instance (New a b p1, New a b p2) => New a b (p1 :& p2) where
  new xs ys = do
    p1 <- new xs ys
    p2 <- new xs ys
    return (p1 :& p2)

pattern (:&&) :: (Backprop a, Backprop b, Reifies z W)
              => BVar z a -> BVar z b -> BVar z (a :& b)
pattern x :&& y <- (\xy -> (xy ^^. t1, xy ^^. t2)->(x, y))
  where
    (:&&) = BP.isoVar2 (:&) (\case x :& y -> (x, y))
{-# COMPLETE (:&&) #-}


t1 :: Lens (a :& b) (a' :& b) a a'
t1 f (x :& y) = (:& y) <$> f x

t2 :: Lens (a :& b) (a :& b') b b'
t2 f (x :& y) = (x :&) <$> f y
