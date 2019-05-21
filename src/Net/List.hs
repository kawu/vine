{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE ScopedTypeVariables #-}


-- | The module provides a couple of list-related, backprop-enabled operations.


module Net.List
  ( softMax
  , softMaxLog
  , normalize
  ) where


import qualified Numeric.Backprop as BP
import qualified Prelude.Backprop as PB
import           Numeric.Backprop (BVar, Reifies, W)

import qualified Net.Util as U


-- -- | Apply softmax to a list.
-- softMax
--   :: (Reifies s W)
--   => BVar s [Double]
--   -> BVar s [Double]
-- softMax x0 =
--   PB.fmap (/norm) x
--   where
--     -- TODO: the following line can be perhaps implemented more efficiently.
--     -- Have a look at the `Basic` module, where `LBP.vmap'` is used instead of
--     -- `LBP.vmap`.
--     x = PB.fmap exp x0
--     norm = PB.sum x


-- | Apply softmax to a list.
softMax
  :: (Reifies s W)
  => [BVar s Double]
  -> [BVar s Double]
softMax = normalize . map exp


-- | Softmax in log domain (a.k.a. exp-normalize, see
-- https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick)
softMaxLog
  :: forall s. (Reifies s W)
  => [BVar s Double]
  -> [BVar s Double]
softMaxLog xs =
  map (/norm) ys
  where
    b = maximum xs -- :: BVar s Double
    ys = map (exp . (\x -> x - b)) xs
    norm = sum ys


-- | Normalize a list of non-negative values.
normalize
  :: (Reifies s W)
  => [BVar s Double]
  -> [BVar s Double]
-- normalize x =
--   map (/norm) x
--   where
--     norm = sum x
normalize x
  | any (<0) x = error "Net.List: negative element"
  | otherwise = map (/norm) x
  where
    norm = sum x
