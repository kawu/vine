{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}


module Net.Graph2.Marginals
  ( approxMarginals
  , approxMarginals'
  , approxMarginalsC
  , approxMarginalsC'

  , Res(..)
  ) where


import           GHC.Generics (Generic)

import           Control.Monad (guard)

import qualified Data.Map.Strict as M
import qualified Data.Vector.Sized as VS
import qualified Data.Graph as G
import qualified Data.List as List

import qualified Test.SmallCheck.Series as SC

import qualified Numeric.Backprop as BP
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (BVar, Reifies, W, dot)

import qualified Net.List as NL
import           Net.Graph2.BiComp (Pot, Prob, Vec(..), Vec8, Out(..))
import qualified Net.Graph2.BiComp as B
import           Graph (Graph, Arc, incoming, outgoing)


----------------------------------------------
-- Approximate marginal probabilities
--
-- (version without constraints bis)
----------------------------------------------


-- | Approximate marginal probabilities
approxMarginals'
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Arc potentials
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Node potentials
  -> Int 
    -- ^ Depth
  -> M.Map Arc (BVar s (Vec8 Prob))
approxMarginals' graph potMap nodMap k = M.fromList $ do
  arc <- M.keys potMap
  return (arc, approxMarginals1' graph potMap nodMap arc k)


-- | Approximate the marginal probabilities of the given arc.  If @depth = 0@,
-- only the potential of the arc is taken into account.
approxMarginals1'
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Arc potentials
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Node potentials
  -> Arc
    -- ^ The arc in focus
  -> Int 
    -- ^ Depth
  -> BVar s (Vec8 Prob)
approxMarginals1' graph potMap nodMap (v, w) k =
  obfuscateBP . M.fromList $ zip arcValues probs
  where
    probs = NL.normalize $ do
      out <- arcValues
      return
        $ inside' graph potMap nodMap v (depVal out) k
        * outside' graph potMap nodMap (v, w) (hedVal out) k
        * arcPotExp potMap (v, w) out


-- | Inside pass
inside'
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding arc potential map
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map
  -> G.Vertex
    -- ^ The node in question
  -> Bool
    -- ^ The value assigned to the node
  -> Int
    -- ^ Depth (optional?)
  -> BVar s Double
    -- ^ The resulting (sum of) potential(s)
inside' graph potMap nodMap v x k =
  here * down
  where
    here = there nodMap v x
    down
      | k <= 0 = 1
      | otherwise = product $ do
          -- for each child
          c <- incoming v graph
          -- return the corresponding sum
          return . sum $ do
            -- for each corresponding arc value
            y <- arcValues
            -- such that the head of the value = x
            guard $ hedVal y == x
            -- return the resulting (exponential) potential
            return $ arcPotExp potMap (c, v) y
                   * inside' graph potMap nodMap c (depVal y) (k-1)


-- | Outside computation
outside'
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding potential map
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map
  -> Arc
    -- ^ The arc in question @(v, w)@; we focus on the node @w@, the other is
    -- given as context
  -> Bool
    -- ^ The value assigned to the node @w@
  -> Int
    -- ^ Depth (optional?)
  -> BVar s Double
    -- ^ The resulting (sum of) (exp-)potential(s)
outside' graph potMap nodMap (v, w) x k =
  here * both
  where
    here = there nodMap w x
    both
      | k <= 0 = 1
      | otherwise = up * down
    up = product $ do
      -- for each parent (should be only one!)
      p <- outgoing w graph
      return . sum $ do
        -- for each corresponding arc value
        y <- arcValues
        -- such that the dependent of the value = x
        guard $ depVal y == x
        -- return the resulting (exponential) potential
        return $ arcPotExp potMap (w, p) y
               * outside' graph potMap nodMap (w, p) (hedVal y) (k-1)
    down = product $ do
      -- for each child
      c <- incoming w graph
      -- different than v
      guard $ c /= v
      -- return the corresponding sum
      return . sum $ do
        -- for each corresponding arc value
        y <- arcValues
        -- such that the head of the value = x
        guard $ hedVal y == x
        -- return the resulting (exponential) potential
        return $ arcPotExp potMap (c, w) y
               * inside' graph potMap nodMap c (depVal y) (k-1)


-- | Node potential at a given node (TODO: make analogy to `arcPotExp`)
there
  :: (Reifies s W) 
  => M.Map G.Vertex (BVar s Double)
  -> G.Vertex
  -> Bool
  -> BVar s Double
there nodMap u z
  | z = exp (nodMap M.! u)
  | otherwise = 1.0


----------------------------------------------
-- Approximate marginal probabilities
--
-- (version without constraints)
----------------------------------------------


-- | Approximate marginal probabilities
approxMarginals
  :: (Reifies s W)
  => Graph a b
  -> M.Map Arc (BVar s (Vec8 Pot))
  -> Int -- ^ Depth
  -> M.Map Arc (BVar s (Vec8 Prob))
approxMarginals graph potMap k = M.fromList $ do
  arc <- M.keys potMap
  return (arc, approxMarginals1 graph potMap arc k)


-- | Approx the marginal probabilities of the given arc.  If @depth = 0@, only
-- the potential of the arc is taken into account.
approxMarginals1
  :: (Reifies s W)
  => Graph a b
  -> M.Map Arc (BVar s (Vec8 Pot))
  -> Arc
  -> Int -- ^ Depth
  -> BVar s (Vec8 Prob)
  -- -> M.Map (Out Bool) (BVar s Double)
approxMarginals1 graph potMap (v, w) k = 
  obfuscateBP . M.fromList $ zip arcValues pots
  where
    pots = NL.normalize $ do
      out <- arcValues
      return
        $ inside graph potMap v (depVal out) k
        * outside graph potMap (v, w) (hedVal out) k
        * arcPotExp potMap (v, w) out


-- | A semi-lifted version of `obfuscate` in the `Net.Graph2` module.
--
-- TODO: this uses `VS.fromList` which is supposedly very inefficient in case
-- of long lists.  But our list isn't long, is it?
--
-- TODO: relate to `obfuscate` in tests?  Or maybe have a unified
-- implementation?
--
obfuscateBP
  :: forall s p. (Reifies s W)
  => M.Map (Out Bool) (BVar s Double)
  -> BVar s (Vec8 p)
obfuscateBP =
  BP.coerceVar . LBP.vector . from . M.elems
  where
    from = just . VS.fromList :: [a] -> VS.Vector 8 a
    just Nothing =
      error "Net.Graph2.mkVec: got Nothing"
    just (Just x) = x


-- | Outside computation
outside
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding potential map
  -> Arc
    -- ^ The arc in question
  -> Bool
    -- ^ The value assigned to the node
  -> Int
    -- ^ Depth (optional?)
  -> BVar s Double
    -- ^ The resulting (sum of) potential(s)
outside graph potMap (v, w) x k
  | k <= 0 = 1
  | otherwise = up * down
  where
    up = product $ do
      -- for each parent (should be only one!)
      p <- outgoing w graph
      return . sum $ do
        -- for each corresponding arc value
        y <- arcValues
        -- such that the dependent of the value = x
        guard $ depVal y == x
        -- return the resulting (exponential) potential
        return $ arcPotExp potMap (w, p) y
               * outside graph potMap (w, p) (hedVal y) (k-1)
    down = product $ do
      -- for each child
      c <- incoming w graph
      -- different than v
      guard $ c /= v
      -- return the corresponding sum
      return . sum $ do
        -- for each corresponding arc value
        y <- arcValues
        -- such that the head of the value = x
        guard $ hedVal y == x
        -- return the resulting (exponential) potential
        return $ arcPotExp potMap (c, w) y
               * inside graph potMap c (depVal y) (k-1)


-- | Inside pass
inside
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding potential map
  -> G.Vertex
    -- ^ The node in question
  -> Bool
    -- ^ The value assigned to the node
  -> Int
    -- ^ Depth (optional?)
  -> BVar s Double
    -- ^ The resulting (sum of) potential(s)
inside graph potMap v x k
  | k <= 0 = 1
  | otherwise = product $ do
      -- for each child
      c <- incoming v graph
      -- return the corresponding sum
      return . sum $ do
        -- for each corresponding arc value
        y <- arcValues
        -- such that the head of the value = x
        guard $ hedVal y == x
        -- return the resulting (exponential) potential
        return $ arcPotExp potMap (c, v) y
               * inside graph potMap c (depVal y) (k-1)


-- | The list of possible values of an arc
arcValues :: [Out Bool]
arcValues = B.enumerate


-- | Potential of the given arc
arcPotExp
  :: forall s. (Reifies s W)
  => M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Potential map
  -> Arc
    -- ^ The arc in question
  -> Out Bool
    -- ^ The value
  -> BVar s Double
arcPotExp potMap arc out =
  exp $ potVec `dot` BP.constVar (B.mask out)
  where
    potVec = BP.coerceVar (potMap M.! arc)


----------------------------------------------
-- Constrained version bis
----------------------------------------------


-- | Approximate marginal probabilities
approxMarginalsC'
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Arc potentials
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Node potentials
  -> Int
    -- ^ Depth
  -> M.Map Arc (BVar s (Vec8 Prob))
approxMarginalsC' graph potMap nodMap k = M.fromList $ do
  arc <- M.keys potMap
  return (arc, approxMarginalsC1' graph potMap nodMap arc k)


-- | Approx the marginal probabilities of the given arc.  If @depth = 0@, only
-- the potential of the arc is taken into account.
approxMarginalsC1'
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding arc potential map
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map
  -> Arc
    -- ^ The arc in focus
  -> Int 
    -- ^ Depth
  -> BVar s (Vec8 Prob)
approxMarginalsC1' graph potMap nodMap (v, w) k =
  -- TODO: make sure that here also the eventual constraints are handled!
  obfuscateBP . M.fromList $ zip arcValues pots
  where
    pots = NL.normalize $ do
      out <- arcValues
      return
        $ insideC' graph potMap nodMap v (depVal out) (arcVal out) k
        * outsideC' graph potMap nodMap (v, w) (hedVal out) (arcVal out) k
        * arcPotExp potMap (v, w) out

-- | Inside pass, constrained version
insideC'
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding arc potential map
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map
  -> G.Vertex
    -- ^ The node in question
  -> Bool
    -- ^ The value assigned to the node
  -> Bool
    -- ^ The value assigned to the arc from the given node to its parent
  -> Int
    -- ^ Depth (optional?)
  -> BVar s Double
    -- ^ The resulting (sum of) potential(s)
insideC' graph potMap nodMap v x xArc k =
  here * downOne
  where
    here = there nodMap v x
    -- constraints are handled below
    downOne
      | k <= 0 = 1
      | x = zeroTrue down + oneTrue down + moreTrue down
      | not x && not xArc = zeroTrue down + moreTrue down
      | not x && xArc = oneTrue down + moreTrue down
      | otherwise = error "insideC': impossible happened"
--       | otherwise = zeroTrue down + oneTrue down + moreTrue down
    down = product $ do
      -- for each child
      c <- incoming v graph
      -- return the corresponding sum
      return . sum $ do
        -- for each corresponding arc value
        y <- arcValues
        -- such that the head of the value = x
        guard $ hedVal y == x
        -- determine the lower inside value
        let ins = insideC' graph potMap nodMap c (depVal y) (arcVal y) (k-1)
--         -- and ignore it if 0
--         guard $ ins > 0
        -- the resulting (exponential) potential
        let val = arcPotExp potMap (c, v) y * ins
        return $
          if arcVal y
             then justOne val
             else justZero val


-- | Outside computation
outsideC'
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding potential map
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map
  -> Arc
    -- ^ The arc in question
  -> Bool
    -- ^ The value assigned to the head node of the arc
  -> Bool
    -- ^ The value assigned to the arc itself
  -> Int
    -- ^ Depth (optional?)
  -> BVar s Double
    -- ^ The resulting (sum of) potential(s)
outsideC' graph potMap nodMap (v, w) x xArc k =
  here * bothOne
  where
    here = there nodMap w x
    -- constraints are handled below (TODO: the same code as in `insideC'`:
    -- refactor?)
    bothOne
      | k <= 0 = 1
      | x = zeroTrue both + oneTrue both + moreTrue both
      | not x && not xArc = zeroTrue both + moreTrue both
      | not x && xArc = oneTrue both + moreTrue both
      | otherwise = error "outsideC': impossible happened"
--       | otherwise = zeroTrue both + oneTrue both + moreTrue both
    both = up * down
    up = product $ do
      -- for each parent (should be at most one!)
      p <- outgoing w graph
      return . sum $ do
        -- for each corresponding arc value
        y <- arcValues
        -- such that the dependent of the value = x
        guard $ depVal y == x
        -- return the resulting (exponential) potential
        let val = arcPotExp potMap (w, p) y
                * outsideC' graph potMap nodMap (w, p) (hedVal y) (arcVal y) (k-1)
        return $ if arcVal y then justOne val else justZero val
    down = product $ do
      -- for each child
      c <- incoming w graph
      -- different than v
      guard $ c /= v
      -- return the corresponding sum
      return . sum $ do
        -- for each corresponding arc value
        y <- arcValues
        -- such that the head of the value = x
        guard $ hedVal y == x
        -- inside value
        let ins = insideC' graph potMap nodMap c (depVal y) (arcVal y) (k-1)
--         -- ignore if 0
--         guard $ ins > 0
        -- return the resulting (exponential) potential
        let val = arcPotExp potMap (c, w) y * ins
        return $ if arcVal y then justOne val else justZero val


----------------------------------------------
-- Constrained version
----------------------------------------------


-- | Approximate marginal probabilities
approxMarginalsC
  :: (Reifies s W)
  => Graph a b
  -> M.Map Arc (BVar s (Vec8 Pot))
  -> Int -- ^ Depth
  -> M.Map Arc (BVar s (Vec8 Prob))
approxMarginalsC graph potMap k = M.fromList $ do
  arc <- M.keys potMap
  return (arc, approxMarginalsC1 graph potMap arc k)


-- | Approx the marginal probabilities of the given arc.  If @depth = 0@, only
-- the potential of the arc is taken into account.
approxMarginalsC1
  :: (Reifies s W)
  => Graph a b
  -> M.Map Arc (BVar s (Vec8 Pot))
  -> Arc
  -> Int -- ^ Depth
  -> BVar s (Vec8 Prob)
approxMarginalsC1 graph potMap (v, w) k = 
  -- TODO: make sure that here also the eventual constraints are handled!
  obfuscateBP . M.fromList $ zip arcValues pots
  where
    pots = NL.normalize $ do
      out <- arcValues
      return
        $ insideC graph potMap v (depVal out) (arcVal out) k
        * outsideC graph potMap (v, w) (hedVal out) (arcVal out) k
        * arcPotExp potMap (v, w) out


-- | Result depending on the number of `true` incoming arcs.
data Res a = Res
  { zeroTrue :: a
    -- ^ All the incoming arcs are `False`.
  , oneTrue  :: a
    -- ^ One of the incoming arcs is `True`.
  , moreTrue :: a
    -- ^ More than one of the incoming arcs is `True`.
  } deriving (Generic, Show, Eq, Ord)

-- Allows to use SmallCheck.
instance (SC.Serial m a) => SC.Serial m (Res a)

instance Num a => Num (Res a) where
  r1 * r2 = Res
    { zeroTrue =
        zeroTrue r1 * zeroTrue r2
    , oneTrue = List.foldl1' (+)
        [ zeroTrue r1 * oneTrue  r2
        , oneTrue  r1 * zeroTrue r2
        ]
    , moreTrue = List.foldl1' (+)
        [ zeroTrue r1 * moreTrue r2
        , moreTrue r1 * zeroTrue r2
        , oneTrue  r1 * oneTrue  r2
        , oneTrue  r1 * moreTrue r2
        , moreTrue r1 * oneTrue  r2
        , moreTrue r1 * moreTrue r2
        ]
    }
  r1 + r2 = Res
    { zeroTrue = zeroTrue r1 + zeroTrue r2
    , oneTrue  = oneTrue  r1 + oneTrue  r2
    , moreTrue = moreTrue r1 + moreTrue r2
    }
  negate (Res x y z) = Res (negate x) (negate y) (negate z)
  abs (Res x y z) = Res (abs x) (abs y) (abs z)
  signum (Res x y z) = Res (signum x) (signum y) (signum z)
  -- fromInteger x = Res (fromInteger x) (fromInteger x) (fromInteger x)
  fromInteger x = Res (fromInteger x) 0 0


justZero :: Num a => a -> Res a
justZero x = Res
  { zeroTrue = x
  , oneTrue  = 0
  , moreTrue = 0
  }


justOne :: Num a => a -> Res a
justOne x = Res
  { zeroTrue = 0
  , oneTrue  = x
  , moreTrue = 0
  }


-- | Inside pass, constrained version
insideC
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding potential map
  -> G.Vertex
    -- ^ The node in question
  -> Bool
    -- ^ The value assigned to the node
  -> Bool
    -- ^ The value assigned to the arc from the given node to its parent
  -> Int
    -- ^ Depth (optional?)
  -> BVar s Double
    -- ^ The resulting (sum of) potential(s)
insideC graph potMap v x xArc k
  | k <= 0 = 1
--   | otherwise = zeroTrue down + oneTrue down + moreTrue down
--   | null (incoming v graph) =
--       case (x, xArc) of
--         (True,   True) -> 1
--         (False,  True) -> 0
--         (True,  False) -> 1
--         (False, False) -> 1
  | x = zeroTrue down + oneTrue down + moreTrue down
  | not x && not xArc = zeroTrue down + moreTrue down
  | not x && xArc = oneTrue down + moreTrue down
  | otherwise = error "insideC: impossible happened"
  where
    down = product $ do
      -- for each child
      c <- incoming v graph
      -- return the corresponding sum
      return . sum $ do
        -- for each corresponding arc value
        y <- arcValues
        -- such that the head of the value = x
        guard $ hedVal y == x
        -- determine the lower inside value
        let ins = insideC graph potMap c (depVal y) (arcVal y) (k-1)
--         -- and ignore it if 0
--         guard $ ins > 0
        -- the resulting (exponential) potential
        let val = arcPotExp potMap (c, v) y * ins
        return $
          if arcVal y
             then justOne val
             else justZero val


-- | Outside computation (TODO: not yet constrained!)
outsideC
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding potential map
  -> Arc
    -- ^ The arc in question
  -> Bool
    -- ^ The value assigned to the head node of the arc
  -> Bool
    -- ^ The value assigned to the arc itself
  -> Int
    -- ^ Depth (optional?)
  -> BVar s Double
    -- ^ The resulting (sum of) potential(s)
outsideC graph potMap (v, w) x xArc k
  | k <= 0 = 1
--   | otherwise = zeroTrue both + oneTrue both + moreTrue both
  | x = zeroTrue both + oneTrue both + moreTrue both
  | not x && not xArc = zeroTrue both + moreTrue both
  | not x && xArc = oneTrue both + moreTrue both
  | otherwise = error "outsideC: impossible happened"
  where
    both = up * down
    up = product $ do
      -- for each parent (should be at most one!)
      p <- outgoing w graph
      return . sum $ do
        -- for each corresponding arc value
        y <- arcValues
        -- such that the dependent of the value = x
        guard $ depVal y == x
        -- return the resulting (exponential) potential
        let val = arcPotExp potMap (w, p) y
                * outsideC graph potMap (w, p) (hedVal y) (arcVal y) (k-1)
        return $ if arcVal y then justOne val else justZero val
    down = product $ do
      -- for each child
      c <- incoming w graph
      -- different than v
      guard $ c /= v
      -- return the corresponding sum
      return . sum $ do
        -- for each corresponding arc value
        y <- arcValues
        -- such that the head of the value = x
        guard $ hedVal y == x
        -- inside value
        let ins = insideC graph potMap c (depVal y) (arcVal y) (k-1)
--         -- ignore if 0
--         guard $ ins > 0
        -- return the resulting (exponential) potential
        let val = arcPotExp potMap (c, w) y * ins
        return $ if arcVal y then justOne val else justZero val


----------------------------------------------
-- Log
----------------------------------------------
-- 
-- 
-- -- | Potential of the given arc
-- arcPotExpLog
--   :: forall s. (Reifies s W)
--   => M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ Potential map
--   -> Arc
--     -- ^ The arc in question
--   -> Out Bool
--     -- ^ The value
--   -> BVar s Double
-- arcPotExpLog potMap arc out =
--   potVec `dot` BP.constVar (B.mask out)
--   where
--     potVec = BP.coerceVar (potMap M.! arc)
