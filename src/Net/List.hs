{-# LANGUAGE FlexibleContexts #-}


module Net.List
  ( softmax
  -- , dot
  ) where


import qualified Numeric.Backprop as BP
import qualified Prelude.Backprop as PB
import           Numeric.Backprop (BVar, Reifies, W)


-- | Apply softmax to a list.
softmax
  :: (Reifies s W)
  => BVar s [Double]
  -> BVar s [Double]
softmax x0 =
  PB.fmap (/norm) x
  where
    -- TODO: the following line can be perhaps implemented more efficiently.
    -- Have a look at the `Basic` module, where `LBP.vmap'` is used instead of
    -- `LBP.vmap`.
    x = PB.fmap exp x0
    norm = PB.sum x


-- -- | Dot product.
-- --
-- -- WARNING: the two input list should be of the same size; 
-- -- otherwise, this does not work correctly!
-- --
-- dot 
--   :: (Reifies s W)
--   => BVar s [Double]
--   -> BVar s [Double]
--   -> BVar s [Double]
-- dot = BP.liftOp2 . BP.op2 $ \xs ys ->
--   ( xs `_dot` ys
--   , undefined
--   -- IDEA: , \d -> (d `_dot` ys,  d `_dot` xs)
--   )
-- -- -- ORIGINAL:
-- -- dot = BP.liftOp2 . BP.op2 $ \x y ->
-- --     ( x `H.dot` y
-- --     , \d -> let d' = H.konst d
-- --             in  (d' * y, x * d')
-- --     )
-- {-# INLINE dot #-}
-- 
-- 
-- -- | The actual dot product operation.
-- _dot :: [Double] -> [Double] -> [Double]
-- _dot = undefined
-- {-# INLINE _dot #-}
