{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Calculation of marginal probabilities


module Net.Graph.Marginals
  ( 
    -- * Types
    Res(..)
  , MVar(..)

    -- * Marginals
  , marginalsMemo
  , marginalsMemoC
  , dummyMarginals

    -- * Inside
  , insideLogMemo
  , insideLogMemoC
  ) where


import           GHC.Generics (Generic)

import           Control.Monad (guard)

import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.List as List
import qualified Data.Maybe as Maybe

import qualified Test.SmallCheck.Series as SC

import qualified Numeric.Backprop as BP
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (BVar, Reifies, W, dot)

import qualified Data.MemoCombinators as Memo

import qualified Net.Util as U
import qualified Net.List as NL
import           Net.Graph.Core
import           Net.Graph.Arc (Pot, Prob, Vec(..), Vec8, Out(..))
import qualified Net.Graph.Arc as B
import           Graph (Graph, Arc, incoming, outgoing)


----------------------------------------------
-- Marginal scores
----------------------------------------------


-- | Calculate the marginal scores for all the arcs in the graph.
marginalsMemo
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Arc potentials
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Node potentials
  -> M.Map Arc (BVar s (Vec8 Pot))
marginalsMemo graph potMap nodMap = M.fromList $ do
  arc <- M.keys potMap
  return (arc, marginals arc)
  where
    marginals = marginals1Memo graph potMap nodMap


-- | Calculate the marginal scores of the different labelling combinations for
-- the given arc.
--
-- NOTE: We calculate "marginal scores" rather than marginal probabilities,
-- hence the type of the result: @Vec8 Pot@.  To get the actual probabilities,
-- softmax needs to be applied (we apply it within the context of cross-entropy
-- calculation).
--
marginals1Memo
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Arc potentials
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Node potentials
  -> Arc
    -- ^ The arc in focus
  -> BVar s (Vec8 Pot)
marginals1Memo graph potMap nodMap =
  marginals
  where
    inside = insideLogMemo graph potMap nodMap
    outside = outsideLogMemo graph potMap nodMap inside
    marginals (v, w) =
      B.obfuscateBP $ M.fromList scores
      where
        scores = do
          out <- arcValues
          let val = inside v (depVal out) `mulLog`
                    outside (v, w) (hedVal out) `mulLog`
                    arcPot potMap (v, w) out
          return (out, val)


----------------------------------------------
-- Marginal scores with constraints
----------------------------------------------


-- | Calculate the marginal scores for all the arcs.  Constrained version of
-- `marginalsMemo`.
marginalsMemoC
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Arc potentials
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Node potentials
  -> M.Map Arc (BVar s (Vec8 Pot))
marginalsMemoC graph potMap nodMap = M.fromList $ do
  arc <- M.keys potMap
  return (arc, marginals arc)
  where
    marginals = marginals1MemoC graph potMap nodMap


-- | Calculate the marginal scores for the given arc.  Constrained version of
-- `marginals1Memo`.
marginals1MemoC
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Arc potentials
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Node potentials
  -> Arc
    -- ^ The arc in focus
  -> BVar s (Vec8 Pot)
marginals1MemoC graph potMap nodMap =
  marginals
  where
    inside = insideLogMemoC graph potMap nodMap
    outside = outsideLogMemoC graph potMap nodMap inside
    marginals (v, w) =
      B.obfuscateBP $ M.fromList scores
      where
        fromJust may =
          case may of
            Just x -> x
            -- NOTE: Instead of negative infinitive, we use a very small
            -- number.  Otherwise, numerical problems arise.
            Nothing -> -1.0e100
        scores = do
          out <- arcValues
          let val = fromJust . unMVar $
                inside v (depVal out) (arcVal out) *
                outside (v, w) (hedVal out) (arcVal out) *
                (MVar . Just $ arcPot potMap (v, w) out)
          return (out, val)


----------------------------------------------
-- Dummy marginals
--
-- TODO: comment out, since experimental
----------------------------------------------


-- | Dummy marginals (just softmax?)
dummyMarginals
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Arc potentials
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Node potentials
  -> M.Map Arc (BVar s (Vec8 Pot))
dummyMarginals graph potMap nodMap = M.fromList $ do
  arc <- M.keys potMap
  return (arc, marginals arc)
  where
    marginals = dummyMarginals1 graph potMap nodMap


-- | TODO: WARNING, this completely ignore node potentials!
dummyMarginals1
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Arc potentials
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Node potentials
  -> Arc
    -- ^ The arc in focus
  -> BVar s (Vec8 Pot)
dummyMarginals1 graph potMap nodMap =
  marginals
  where
    marginals (v, w)
      | nodMap M.! v /= 0.0 =
          error "dummyMarginals1: v node potential /= 0"
      | nodMap M.! w /= 0.0 =
          error "dummyMarginals1: v node potential /= 0"
      | otherwise =
          potMap M.! (v, w)


----------------------------------------------
-- Inside-outside
----------------------------------------------


-- | Inside pass (log domain)
--
-- TODO: Consider returning `LVar`.
--
insideLogMemo
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding arc potential map (normal domain!)
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map (normal domain!)
  -> G.Vertex
    -- ^ The node in question
  -> Bool
    -- ^ The value assigned to the node
  -> BVar s Double
    -- ^ The resulting (sum of) potential(s) (log domain!)
insideLogMemo graph potMap nodMap =
  \v -> unLVar . inside v
  where
    inside = Memo.memo2 Memo.integral Memo.bool inside'
    inside' v x =
      hereLog * downLog
      where
        hereLog = LVar $ thereLog nodMap v x
        downLog = product $ do
          -- for each child
          c <- incoming v graph
          -- return the corresponding sum
          return . sum $ do
            -- for each corresponding arc value
            y <- arcValues
            -- such that the head of the value = x
            guard $ hedVal y == x
            -- return the resulting (exponential) potential
            return $ LVar (arcPot potMap (c, v) y) * inside c (depVal y)


-- | Outside computation (log domain)
--
-- TODO: Consider returning `LVar`.
--
outsideLogMemo
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding potential map (normal domain!)
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map (normal domain!)
  -> (G.Vertex -> Bool -> BVar s Double)
    -- ^ Inside calculation function
  -> Arc
    -- ^ The arc in question @(v, w)@; we focus on the node @w@, the other is
    -- given as context
  -> Bool
    -- ^ The value assigned to the node @w@
  -> BVar s Double
    -- ^ The resulting (sum of) (exp-)potential(s) (log domain!)
outsideLogMemo graph potMap nodMap inside =
  outside
  where
    outside = Memo.memo2
      (Memo.pair Memo.integral Memo.integral) Memo.bool outside'
    outside' (v, w) x =
      hereLog `mulLog` bothLog
      where
        hereLog = thereLog nodMap w x
        bothLog = upLog `mulLog` downLog
        upLog = productLog $ do
          -- for each parent (should be only one!)
          p <- outgoing w graph
          return . sumLog $ do
            -- for each corresponding arc value
            y <- arcValues
            -- such that the dependent of the value = x
            guard $ depVal y == x
            -- return the resulting (exponential) potential
            return $ arcPot potMap (w, p) y `mulLog`
                     outside (w, p) (hedVal y)
        downLog = productLog $ do
          -- for each child
          c <- incoming w graph
          -- different than v
          guard $ c /= v
          -- return the corresponding sum
          return . sumLog $ do
            -- for each corresponding arc value
            y <- arcValues
            -- such that the head of the value = x
            guard $ hedVal y == x
            -- return the resulting (exponential) potential
            return $ arcPot potMap (c, w) y `mulLog`
                     inside c (depVal y)


----------------------------------------------
-- Inside-outside with constraints
----------------------------------------------


-- | Inside pass, constrained version
insideLogMemoC
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
  -> MVar s Double
    -- ^ The resulting (sum of) potential(s)
insideLogMemoC graph potMap nodMap =
  \v x xArc -> inside v x xArc
  where
    -- NOTE: memoization on the first two arguments is more efficient?
    inside = Memo.memo2 Memo.integral Memo.bool inside'
    -- inside = Memo.memo3 Memo.integral Memo.bool Memo.bool inside'
    inside' v x =
      -- \xArc -> here * downOne xArc
      \xArc -> here * constrain down x xArc
      where
        here = MVar . Just $ thereLog nodMap v x
        -- constraints are handled below
--         downOne xArc
--           | x = sum
--               [zeroTrue down, oneTrue down, moreTrue down]
--           | not x && not xArc = sum
--               [zeroTrue down, moreTrue down]
--           | not x && xArc = sum
--               [oneTrue down, moreTrue down]
--           | otherwise = error "insideLogMemoC: impossible happened"
-- --           | otherwise = sum
-- --               [zeroTrue down, oneTrue down, moreTrue down]
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
            let ins = inside c (depVal y) (arcVal y)
            -- the resulting (exponential) potential
            let val = ins * (MVar . Just) (arcPot potMap (c, v) y)
            return $ if arcVal y then justOne val else justZero val


-- | Outside computation (log domain)
outsideLogMemoC
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding potential map (normal domain!)
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map (normal domain!)
  -> (G.Vertex -> Bool -> Bool -> MVar s Double)
    -- ^ Inside calculation function
  -> Arc
    -- ^ The arc in question @(v, w)@; we focus on the node @w@, the other is
    -- given as context
  -> Bool
    -- ^ The value assigned to the node @w@
  -> Bool
    -- ^ The value assigned to the arc @(v, w)@
  -> MVar s Double
    -- ^ The resulting (sum of) (exp-)potential(s) (log domain!)
outsideLogMemoC graph potMap nodMap inside =
  outside
  where
    outside = Memo.memo3
      (Memo.pair Memo.integral Memo.integral) Memo.bool Memo.bool outside'
    outside' (v, w) x xArc =
      here * constrain both x xArc
      where
        here = MVar . Just $ thereLog nodMap w x
        both = up * down
        up = product $ do
          -- for each parent (should be only one!)
          p <- outgoing w graph
          return . sum $ do
            -- for each corresponding arc value
            y <- arcValues
            -- such that the dependent of the value = x
            guard $ depVal y == x
            -- return the resulting (exponential) potential
            let val = (MVar . Just $ arcPot potMap (w, p) y)
                    * outside (w, p) (hedVal y) (arcVal y)
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
            -- return the resulting (exponential) potential
            let val = (MVar . Just $ arcPot potMap (c, w) y)
                    * inside c (depVal y) (arcVal y)
            return $ if arcVal y then justOne val else justZero val


-- | Constrain the result for the given node and its surrounding, given the
-- node's label and the label of a specific arc in focus.
constrain 
  :: (Num a)
  => Res a
  -> Bool -- ^ Node label
  -> Bool -- ^ Arc label
  -> a
constrain res x xArc
  | x = sum
      [zeroTrue res, oneTrue res, moreTrue res]
  | not x && not xArc = sum
      [zeroTrue res, moreTrue res]
  | not x && xArc = sum
      [oneTrue res, moreTrue res]
  | otherwise = error "constrain: impossible happened"
{-# INLINE constrain #-}


----------------------------------------------
-- Result
----------------------------------------------


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
    { zeroTrue = zeroTrue r1 * zeroTrue r2
    , oneTrue = sum
        [ zeroTrue r1 * oneTrue  r2
        , oneTrue  r1 * zeroTrue r2
        ]
    , moreTrue = sum
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
  fromInteger x = Res (fromInteger x) 0 0


-- sumMaybes :: (Num a) => [Maybe a] -> Maybe a
-- sumMaybes xs = 
--   case Maybe.catMaybes xs of
--     [] -> Nothing
--     ys -> Just (sum ys)


justZero :: Num a => a -> Res a
justZero x = Res
  -- { zeroTrue = Just x
  { zeroTrue = x
  , oneTrue  = 0
  , moreTrue = 0
  }


justOne :: Num a => a -> Res a
justOne x = Res
  { zeroTrue = 0
  -- , oneTrue  = Just x
  , oneTrue  = x
  , moreTrue = 0
  }


-- justZero :: Num a => a -> Res a
-- justZero x = Res
--   { zeroTrue = Just x
--   , oneTrue  = Nothing
--   , moreTrue = Nothing
--   }
-- 
-- 
-- justOne :: Num a => a -> Res a
-- justOne x = Res
--   { zeroTrue = Nothing
--   , oneTrue  = Just x
--   , moreTrue = Nothing
--   }


-- -- | Result depending on the number of `true` incoming arcs.
-- data Res a = Res
--   { zeroTrue :: Maybe a
--     -- ^ All the incoming arcs are `False`.
--   , oneTrue  :: Maybe a
--     -- ^ One of the incoming arcs is `True`.
--   , moreTrue :: Maybe a
--     -- ^ More than one of the incoming arcs is `True`.
--   } deriving (Generic, Show, Eq, Ord)
-- 
-- -- Allows to use SmallCheck.
-- instance (SC.Serial m a) => SC.Serial m (Res a)
-- 
-- instance Num a => Num (Res a) where
--   r1 * r2 = Res
--     { zeroTrue =
--         (*) <$> zeroTrue r1 <*> zeroTrue r2
--     , oneTrue = sumMaybes
--         [ (*) <$> zeroTrue r1 <*> oneTrue  r2
--         , (*) <$> oneTrue  r1 <*> zeroTrue r2
--         ]
--     , moreTrue = sumMaybes
--         [ (*) <$> zeroTrue r1 <*> moreTrue r2
--         , (*) <$> moreTrue r1 <*> zeroTrue r2
--         , (*) <$> oneTrue  r1 <*> oneTrue  r2
--         , (*) <$> oneTrue  r1 <*> moreTrue r2
--         , (*) <$> moreTrue r1 <*> oneTrue  r2
--         , (*) <$> moreTrue r1 <*> moreTrue r2
--         ]
--     }
--   r1 + r2 = Res
--     { zeroTrue = sumMaybes [zeroTrue r1, zeroTrue r2]
--     , oneTrue  = sumMaybes [oneTrue  r1, oneTrue  r2]
--     , moreTrue = sumMaybes [moreTrue r1, moreTrue r2]
--     }
--   negate (Res x y z) = Res (negate <$> x) (negate <$> y) (negate <$> z)
--   abs (Res x y z) = Res (abs <$> x) (abs <$> y) (abs <$> z)
--   signum (Res x y z) = Res (signum <$> x) (signum <$> y) (signum <$> z)
--   fromInteger x 
--     | x == 0  = Res Nothing Nothing Nothing
--     | otherwise = Res (Just $ fromInteger x) Nothing Nothing


----------------------------------------------
-- Log BVar
----------------------------------------------


-- | Negative infinity
negativeInfinity :: Floating a => a
negativeInfinity = negate (1/0)
{-# INLINE negativeInfinity #-}


-- | log (1 + x)
log1p :: Floating a => a -> a
log1p x = log (1 + x)
{-# INLINE log1p #-}


-- | BVar in log domain
newtype LVar s a = LVar {unLVar :: BVar s a}


instance (Reifies s W, Ord a, Floating a) => Num (LVar s a) where
    (*) (LVar x) (LVar y) = LVar (x+y)
    (+) (LVar x) (LVar y)
        | x >= y    = LVar (x + log1p (exp (y - x)))
        | otherwise = LVar (y + log1p (exp (x - y)))
    (-) (LVar x) (LVar y) =
        LVar (x + log1p (negate (exp (y - x))))
    signum (LVar x) = 1
    negate _    = error "negate LVar"
    abs         = id
    fromInteger = LVar . log . fromInteger


-- | Maybe BVar in log domain; `Nothing` repesents negative infinity.
newtype MVar s a = MVar {unMVar :: Maybe (BVar s a)}


instance (Reifies s W, Ord a, Floating a) => Num (MVar s a) where
  MVar mx * MVar my =
    case (mx, my) of
      (Just x, Just y) -> MVar (Just $ x+y)
      _ -> MVar Nothing
  MVar mx + MVar my =
    case (mx, my) of
      (Just x, Just y) ->
        if x >= y
           then (MVar . Just) (x + log1p (exp (y - x)))
           else (MVar . Just) (y + log1p (exp (x - y)))
      (Just x, Nothing) -> MVar (Just x)
      (Nothing, Just y) -> MVar (Just y)
      _ -> MVar Nothing
  MVar mx - MVar my =
    case (mx, my) of
      (Just x, Just y)  -> (MVar . Just) (x + log1p (negate (exp (y - x))))
      (Just x, Nothing) -> MVar (Just x)
      _ -> error "MVar.1"
  signum (MVar mx) =
    case mx of
      Nothing -> 0
      Just _ -> 1
  negate _    = error "negate MVar"
  abs = id
  fromInteger x
    | x > 0  = MVar . Just . log . fromInteger $ x
    | x == 0 = MVar Nothing
    | otherwise = error "MVar.fromInteger <0"
