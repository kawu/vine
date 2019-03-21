{-# LANGUAGE TupleSections #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE ScopedTypeVariables #-}


-- | The module provies the class `New` for creating new, random networks.
-- Normally we would like to just use MonadRandom, but we need to make the set
-- of labels available at the time of network creation.


module Net.New
  ( New(..)
  , newMap
  ) where


import           GHC.TypeNats (KnownNat, natVal)

import           Control.Monad (forM)

import           System.Random (randomRIO)

import           Data.Proxy (Proxy(..))

import           Numeric.LinearAlgebra.Static.Backprop (R, L)
import qualified Numeric.LinearAlgebra.Static as LA

import qualified Data.Map.Strict as M
import qualified Data.Set as S

import           Net.Util (randomList)


-- | A class of networks which can be randomly created based on the set of node
-- and arc labels.
class New a b p where
  new
    :: S.Set a
      -- ^ Set of node labels
    -> S.Set b
      -- ^ Set of arc labels
    -> IO p

instance New a b Double where
  new _ _ = randomRIO (-0.01, 0.01)

instance (KnownNat n) => New a b (R n) where
  new _ _ = LA.vector <$> randomList n
    where
      n = proxyVal (Proxy :: Proxy n)
      proxyVal = fromInteger . toInteger . natVal

instance (KnownNat n, KnownNat m) => New a b (L n m) where
  new _ _ = LA.matrix <$> randomList (n*m)
    where
      n = proxyVal (Proxy :: Proxy n)
      m = proxyVal (Proxy :: Proxy m)
      proxyVal = fromInteger . toInteger . natVal


-- | Create a new, random map.
newMap
  :: (Ord k, New a b v)
  => S.Set k
  -> S.Set a
  -> S.Set b
  -> IO (M.Map k v)
newMap keySet xs ys =
  fmap M.fromList .
    forM (S.toList keySet) $ \key -> do
      (key,) <$> new xs ys
