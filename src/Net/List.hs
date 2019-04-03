{-# LANGUAGE FlexibleContexts #-}


module Net.List
  ( softmax
  , normalize
  ) where


import qualified Numeric.Backprop as BP
import qualified Prelude.Backprop as PB
import           Numeric.Backprop (BVar, Reifies, W)


-- -- | Apply softmax to a list.
-- softmax
--   :: (Reifies s W)
--   => BVar s [Double]
--   -> BVar s [Double]
-- softmax x0 =
--   PB.fmap (/norm) x
--   where
--     -- TODO: the following line can be perhaps implemented more efficiently.
--     -- Have a look at the `Basic` module, where `LBP.vmap'` is used instead of
--     -- `LBP.vmap`.
--     x = PB.fmap exp x0
--     norm = PB.sum x


-- | Apply softmax to a list (alternative version).
softmax
  :: (Reifies s W)
  => [BVar s Double]
  -> [BVar s Double]
softmax = normalize . map exp


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
