{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}


module Net.Graph2.Marginals
  ( 
  --   approxMarginals
  --   approxMarginals'
    approxMarginalsMemo
  , approxMarginalsMemoC
  -- , approxMarginalsC
  -- , approxMarginalsC'
  , dummyMarginals

  , Res(..)

  -- , insideLog
  , insideLogMemo
  , insideLogMemoC
  , MVar(..)
  ) where


import           GHC.Generics (Generic)

import           Control.Monad (guard)

import qualified Data.Map.Strict as M
import qualified Data.Vector.Sized as VS
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
import           Net.Graph2.BiComp (Pot, Prob, Vec(..), Vec8, Out(..))
import qualified Net.Graph2.BiComp as B
import           Graph (Graph, Arc, incoming, outgoing)


----------------------------------------------
-- Utils
----------------------------------------------
 

-- | The list of possible values of an arc
arcValues :: [Out Bool]
arcValues = B.enumerate


-- -- | Potential of the given arc
-- _arcPot
--   :: forall s. (Reifies s W)
--   => M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ Potential map
--   -> Arc
--     -- ^ The arc in question
--   -> Out Bool
--     -- ^ The value
--   -> BVar s Double
-- _arcPot potMap arc out =


-- | Potential of the given arc
arcPot
  :: forall s. (Reifies s W)
  => M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Potential map
  -> Arc
    -- ^ The arc in question
  -> Out Bool
    -- ^ The value
  -> BVar s Double
arcPot potMap arc out =
  potVec `dot` BP.constVar (B.mask out)
  where
    potVec = BP.coerceVar (potMap M.! arc)
{-# INLINE arcPot #-}


-- | (Exp-)potential of the given arc  (exp . arcPot)
arcPotExp
  :: forall s. (Reifies s W)
  => M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Potential map
  -> Arc
    -- ^ The arc in question
  -> Out Bool
    -- ^ The value
  -> BVar s Double
arcPotExp potMap arc =
  exp . arcPot potMap arc


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


----------------------------------------------
-- Approximate marginal probabilities
--
-- (version with and without constraints bis)
----------------------------------------------


-- -- | Approximate marginal probabilities
-- approxMarginals'
--   :: (Reifies s W)
--   => Graph a b
--     -- ^ The underlying graph
--   -> M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ Arc potentials
--   -> M.Map G.Vertex (BVar s Double)
--     -- ^ Node potentials
--   -> Int
--     -- ^ Depth
--   -> M.Map Arc (BVar s (Vec8 Pot))
-- approxMarginals' graph potMap nodMap k = M.fromList $ do
--   arc <- M.keys potMap
--   return (arc, approxMarginals1' graph potMap nodMap arc k)


-- | Approximate marginal probabilities with constraints
--
-- TODO: are the output really potentials, or just probabilities in log domain?
--
approxMarginalsMemoC
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Arc potentials
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Node potentials
  -> M.Map Arc (BVar s (Vec8 Pot))
approxMarginalsMemoC graph potMap nodMap = M.fromList $ do
  arc <- M.keys potMap
  return (arc, marginals arc)
  where
    marginals = approxMarginals1MemoC graph potMap nodMap


-- | Approximate the marginal probabilities of the given arc.  If @depth = 0@,
-- only the potential of the arc is taken into account.
approxMarginals1MemoC
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
approxMarginals1MemoC graph potMap nodMap =
  marginals
  where
    inside = insideLogMemoC graph potMap nodMap
    outside = outsideLogMemoC graph potMap nodMap inside
    marginals (v, w) =
      obfuscateBP . M.fromList $ zip arcValues probs
      where
        fromJust may =
          case may of
            Just x -> x
            -- TODO: is it a proper choice of epsilon?
            Nothing -> -1.0e100
        probs = do
          out <- arcValues
          return . fromJust . unMVar $
            inside v (depVal out) (arcVal out) *
            outside (v, w) (hedVal out) (arcVal out) *
            (MVar . Just $ arcPot potMap (v, w) out)


-- | Approximate marginal probabilities
--
-- TODO: are the output really potentials, or just probabilities in log domain?
-- TODO: this is no longer an approximation!
--
approxMarginalsMemo
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ Arc potentials
  -> M.Map G.Vertex (BVar s Double)
    -- ^ Node potentials
  -> M.Map Arc (BVar s (Vec8 Pot))
approxMarginalsMemo graph potMap nodMap = M.fromList $ do
  arc <- M.keys potMap
  return (arc, marginals arc)
  where
    marginals = approxMarginals1Memo graph potMap nodMap


-- | Approximate the marginal probabilities of the given arc.  If @depth = 0@,
-- only the potential of the arc is taken into account.
approxMarginals1Memo
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
approxMarginals1Memo graph potMap nodMap =
  marginals
  where
    inside = insideLogMemo graph potMap nodMap
    outside = outsideLogMemo graph potMap nodMap inside
    marginals (v, w) =
      obfuscateBP . M.fromList $ zip arcValues probs
      where
        -- TODO: probs is not accurate, these are not probabilities but
        -- potential sums, right?
        probs = do
          out <- arcValues
          return $   inside v (depVal out)
            `mulLog` outside (v, w) (hedVal out)
            `mulLog` arcPot potMap (v, w) out


-- -- | Approximate the marginal probabilities of the given arc.  If @depth = 0@,
-- -- only the potential of the arc is taken into account.
-- approxMarginals1'
--   :: (Reifies s W)
--   => Graph a b
--     -- ^ The underlying graph
--   -> M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ Arc potentials
--   -> M.Map G.Vertex (BVar s Double)
--     -- ^ Node potentials
--   -> Arc
--     -- ^ The arc in focus
--   -> Int
--     -- ^ Depth
--   -> BVar s (Vec8 Pot)
-- approxMarginals1' graph potMap nodMap (v, w) k =
--   obfuscateBP . M.fromList $ zip arcValues probs
--   where
--     -- probs = NL.normalize . map exp $ do
--     -- probs = NL.softMaxLog $ do
--     probs = do
--       out <- arcValues
--       return $   insideLogK graph potMap nodMap v (depVal out) k
--         `mulLog` outsideLogK graph potMap nodMap (v, w) (hedVal out) k
--         `mulLog` arcPot potMap (v, w) out


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


-- | Node potential at a given node in log domain
thereLog
  :: (Reifies s W) 
  => M.Map G.Vertex (BVar s Double)
  -> G.Vertex
  -> Bool
  -> BVar s Double
thereLog nodMap u z
  | z = nodMap M.! u
  | otherwise = 0.0


----------------------------------------------
-- Dummy marginals
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
    -- marginals = approxMarginals1Memo graph potMap nodMap
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
-- Constrained version bis
----------------------------------------------


-- -- | Approximate marginal probabilities
-- approxMarginalsC'
--   :: (Reifies s W)
--   => Graph a b
--     -- ^ The underlying graph
--   -> M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ Arc potentials
--   -> M.Map G.Vertex (BVar s Double)
--     -- ^ Node potentials
--   -> Int
--     -- ^ Depth
--   -> M.Map Arc (BVar s (Vec8 Prob))
-- approxMarginalsC' graph potMap nodMap k = M.fromList $ do
--   arc <- M.keys potMap
--   return (arc, approxMarginalsC1' graph potMap nodMap arc k)
-- 
-- 
-- -- | Approx the marginal probabilities of the given arc.  If @depth = 0@, only
-- -- the potential of the arc is taken into account.
-- approxMarginalsC1'
--   :: (Reifies s W)
--   => Graph a b
--     -- ^ The underlying graph
--   -> M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ The corresponding arc potential map
--   -> M.Map G.Vertex (BVar s Double)
--     -- ^ The corresponding node potential map
--   -> Arc
--     -- ^ The arc in focus
--   -> Int 
--     -- ^ Depth
--   -> BVar s (Vec8 Prob)
-- approxMarginalsC1' graph potMap nodMap (v, w) k =
--   -- TODO: make sure that here also the eventual constraints are handled!
--   obfuscateBP . M.fromList $ zip arcValues pots
--   where
--     pots = NL.normalize $ do
--       out <- arcValues
--       return
--         $ insideC' graph potMap nodMap v (depVal out) (arcVal out) k
--         * outsideC' graph potMap nodMap (v, w) (hedVal out) (arcVal out) k
--         * arcPotExp potMap (v, w) out
-- 
-- -- | Inside pass, constrained version
-- insideC'
--   :: (Reifies s W)
--   => Graph a b
--     -- ^ The underlying graph
--   -> M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ The corresponding arc potential map
--   -> M.Map G.Vertex (BVar s Double)
--     -- ^ The corresponding node potential map
--   -> G.Vertex
--     -- ^ The node in question
--   -> Bool
--     -- ^ The value assigned to the node
--   -> Bool
--     -- ^ The value assigned to the arc from the given node to its parent
--   -> Int
--     -- ^ Depth (optional?)
--   -> BVar s Double
--     -- ^ The resulting (sum of) potential(s)
-- insideC' graph potMap nodMap v x xArc k =
--   here * downOne
--   where
--     here = there nodMap v x
--     -- constraints are handled below
--     downOne
--       | k <= 0 = 1
--       | x = zeroTrue down + oneTrue down + moreTrue down
--       | not x && not xArc = zeroTrue down + moreTrue down
--       | not x && xArc = oneTrue down + moreTrue down
--       | otherwise = error "insideC': impossible happened"
-- --       | otherwise = zeroTrue down + oneTrue down + moreTrue down
--     down = product $ do
--       -- for each child
--       c <- incoming v graph
--       -- return the corresponding sum
--       return . sum $ do
--         -- for each corresponding arc value
--         y <- arcValues
--         -- such that the head of the value = x
--         guard $ hedVal y == x
--         -- determine the lower inside value
--         let ins = insideC' graph potMap nodMap c (depVal y) (arcVal y) (k-1)
-- --         -- and ignore it if 0
-- --         guard $ ins > 0
--         -- the resulting (exponential) potential
--         let val = arcPotExp potMap (c, v) y * ins
--         return $
--           if arcVal y
--              then justOne val
--              else justZero val
-- 
-- 
-- -- | Outside computation
-- outsideC'
--   :: (Reifies s W)
--   => Graph a b
--     -- ^ The underlying graph
--   -> M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ The corresponding potential map
--   -> M.Map G.Vertex (BVar s Double)
--     -- ^ The corresponding node potential map
--   -> Arc
--     -- ^ The arc in question
--   -> Bool
--     -- ^ The value assigned to the head node of the arc
--   -> Bool
--     -- ^ The value assigned to the arc itself
--   -> Int
--     -- ^ Depth (optional?)
--   -> BVar s Double
--     -- ^ The resulting (sum of) potential(s)
-- outsideC' graph potMap nodMap (v, w) x xArc k =
--   here * bothOne
--   where
--     here = there nodMap w x
--     -- constraints are handled below (TODO: the same code as in `insideC'`:
--     -- refactor?)
--     bothOne
--       | k <= 0 = 1
--       | x = zeroTrue both + oneTrue both + moreTrue both
--       | not x && not xArc = zeroTrue both + moreTrue both
--       | not x && xArc = oneTrue both + moreTrue both
--       | otherwise = error "outsideC': impossible happened"
-- --       | otherwise = zeroTrue both + oneTrue both + moreTrue both
--     both = up * down
--     up = product $ do
--       -- for each parent (should be at most one!)
--       p <- outgoing w graph
--       return . sum $ do
--         -- for each corresponding arc value
--         y <- arcValues
--         -- such that the dependent of the value = x
--         guard $ depVal y == x
--         -- return the resulting (exponential) potential
--         let val = arcPotExp potMap (w, p) y
--                 * outsideC' graph potMap nodMap (w, p) (hedVal y) (arcVal y) (k-1)
--         return $ if arcVal y then justOne val else justZero val
--     down = product $ do
--       -- for each child
--       c <- incoming w graph
--       -- different than v
--       guard $ c /= v
--       -- return the corresponding sum
--       return . sum $ do
--         -- for each corresponding arc value
--         y <- arcValues
--         -- such that the head of the value = x
--         guard $ hedVal y == x
--         -- inside value
--         let ins = insideC' graph potMap nodMap c (depVal y) (arcVal y) (k-1)
-- --         -- ignore if 0
-- --         guard $ ins > 0
--         -- return the resulting (exponential) potential
--         let val = arcPotExp potMap (c, w) y * ins
--         return $ if arcVal y then justOne val else justZero val


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


-- | Maybe BVar in log domain
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


----------------------------------------------
-- Result
----------------------------------------------


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



----------------------------------------------
-- Constrained version
----------------------------------------------


-- -- | Approximate marginal probabilities
-- approxMarginalsC
--   :: (Reifies s W)
--   => Graph a b
--   -> M.Map Arc (BVar s (Vec8 Pot))
--   -> Int -- ^ Depth
--   -> M.Map Arc (BVar s (Vec8 Prob))
-- approxMarginalsC graph potMap k = M.fromList $ do
--   arc <- M.keys potMap
--   return (arc, approxMarginalsC1 graph potMap arc k)
-- 
-- 
-- -- | Approx the marginal probabilities of the given arc.  If @depth = 0@, only
-- -- the potential of the arc is taken into account.
-- approxMarginalsC1
--   :: (Reifies s W)
--   => Graph a b
--   -> M.Map Arc (BVar s (Vec8 Pot))
--   -> Arc
--   -> Int -- ^ Depth
--   -> BVar s (Vec8 Prob)
-- approxMarginalsC1 graph potMap (v, w) k = 
--   -- TODO: make sure that here also the eventual constraints are handled!
--   obfuscateBP . M.fromList $ zip arcValues pots
--   where
--     pots = NL.normalize $ do
--       out <- arcValues
--       return
--         $ insideC graph potMap v (depVal out) (arcVal out) k
--         * outsideC graph potMap (v, w) (hedVal out) (arcVal out) k
--         * arcPotExp potMap (v, w) out
-- 
-- 
-- -- | Inside pass, constrained version
-- insideC
--   :: (Reifies s W)
--   => Graph a b
--     -- ^ The underlying graph
--   -> M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ The corresponding potential map
--   -> G.Vertex
--     -- ^ The node in question
--   -> Bool
--     -- ^ The value assigned to the node
--   -> Bool
--     -- ^ The value assigned to the arc from the given node to its parent
--   -> Int
--     -- ^ Depth (optional?)
--   -> BVar s Double
--     -- ^ The resulting (sum of) potential(s)
-- insideC graph potMap v x xArc k
--   | k <= 0 = 1
-- --   | otherwise = zeroTrue down + oneTrue down + moreTrue down
-- --   | null (incoming v graph) =
-- --       case (x, xArc) of
-- --         (True,   True) -> 1
-- --         (False,  True) -> 0
-- --         (True,  False) -> 1
-- --         (False, False) -> 1
--   | x = zeroTrue down + oneTrue down + moreTrue down
--   | not x && not xArc = zeroTrue down + moreTrue down
--   | not x && xArc = oneTrue down + moreTrue down
--   | otherwise = error "insideC: impossible happened"
--   where
--     down = product $ do
--       -- for each child
--       c <- incoming v graph
--       -- return the corresponding sum
--       return . sum $ do
--         -- for each corresponding arc value
--         y <- arcValues
--         -- such that the head of the value = x
--         guard $ hedVal y == x
--         -- determine the lower inside value
--         let ins = insideC graph potMap c (depVal y) (arcVal y) (k-1)
-- --         -- and ignore it if 0
-- --         guard $ ins > 0
--         -- the resulting (exponential) potential
--         let val = arcPotExp potMap (c, v) y * ins
--         return $
--           if arcVal y
--              then justOne val
--              else justZero val
-- 
-- 
-- -- | Outside computation (TODO: not yet constrained!)
-- outsideC
--   :: (Reifies s W)
--   => Graph a b
--     -- ^ The underlying graph
--   -> M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ The corresponding potential map
--   -> Arc
--     -- ^ The arc in question
--   -> Bool
--     -- ^ The value assigned to the head node of the arc
--   -> Bool
--     -- ^ The value assigned to the arc itself
--   -> Int
--     -- ^ Depth (optional?)
--   -> BVar s Double
--     -- ^ The resulting (sum of) potential(s)
-- outsideC graph potMap (v, w) x xArc k
--   | k <= 0 = 1
-- --   | otherwise = zeroTrue both + oneTrue both + moreTrue both
--   | x = zeroTrue both + oneTrue both + moreTrue both
--   | not x && not xArc = zeroTrue both + moreTrue both
--   | not x && xArc = oneTrue both + moreTrue both
--   | otherwise = error "outsideC: impossible happened"
--   where
--     both = up * down
--     up = product $ do
--       -- for each parent (should be at most one!)
--       p <- outgoing w graph
--       return . sum $ do
--         -- for each corresponding arc value
--         y <- arcValues
--         -- such that the dependent of the value = x
--         guard $ depVal y == x
--         -- return the resulting (exponential) potential
--         let val = arcPotExp potMap (w, p) y
--                 * outsideC graph potMap (w, p) (hedVal y) (arcVal y) (k-1)
--         return $ if arcVal y then justOne val else justZero val
--     down = product $ do
--       -- for each child
--       c <- incoming w graph
--       -- different than v
--       guard $ c /= v
--       -- return the corresponding sum
--       return . sum $ do
--         -- for each corresponding arc value
--         y <- arcValues
--         -- such that the head of the value = x
--         guard $ hedVal y == x
--         -- inside value
--         let ins = insideC graph potMap c (depVal y) (arcVal y) (k-1)
-- --         -- ignore if 0
-- --         guard $ ins > 0
--         -- return the resulting (exponential) potential
--         let val = arcPotExp potMap (c, w) y * ins
--         return $ if arcVal y then justOne val else justZero val


----------------------------------------------
-- Log
----------------------------------------------


-- | Inside pass (log domain)
insideLogK
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
  -> Int
    -- ^ Depth (optional?)
  -> BVar s Double
    -- ^ The resulting (sum of) potential(s) (log domain!)
insideLogK graph potMap nodMap v x k =
  hereLog `mulLog` downLog
  where
    hereLog = thereLog nodMap v x
    downLog
      | k <= 0 = log 1
      | otherwise = productLog $ do
          -- for each child
          c <- incoming v graph
          -- return the corresponding sum
          return . sumLog $ do
            -- for each corresponding arc value
            y <- arcValues
            -- such that the head of the value = x
            guard $ hedVal y == x
            -- return the resulting (exponential) potential
            return $
              arcPot potMap (c, v) y `mulLog`
              insideLogK graph potMap nodMap c (depVal y) (k-1)


-- | Outside computation (log domain)
outsideLogK
  :: (Reifies s W)
  => Graph a b
    -- ^ The underlying graph
  -> M.Map Arc (BVar s (Vec8 Pot))
    -- ^ The corresponding potential map (normal domain!)
  -> M.Map G.Vertex (BVar s Double)
    -- ^ The corresponding node potential map (normal domain!)
  -> Arc
    -- ^ The arc in question @(v, w)@; we focus on the node @w@, the other is
    -- given as context
  -> Bool
    -- ^ The value assigned to the node @w@
  -> Int
    -- ^ Depth (optional?)
  -> BVar s Double
    -- ^ The resulting (sum of) (exp-)potential(s) (log domain!)
outsideLogK graph potMap nodMap (v, w) x k =
  hereLog `mulLog` bothLog
  where
    hereLog = thereLog nodMap w x
    bothLog
      | k <= 0 = log 1
      | otherwise = upLog `mulLog` downLog
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
                 outsideLogK graph potMap nodMap (w, p) (hedVal y) (k-1)
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
                 insideLogK graph potMap nodMap c (depVal y) (k-1)


----------------------------------------------
-- Log with no depth constraint
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


-- -- | Inside pass (log domain)
-- insideLogMemo
--   :: (Reifies s W)
--   => Graph a b
--     -- ^ The underlying graph
--   -> M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ The corresponding arc potential map (normal domain!)
--   -> M.Map G.Vertex (BVar s Double)
--     -- ^ The corresponding node potential map (normal domain!)
--   -> G.Vertex
--     -- ^ The node in question
--   -> Bool
--     -- ^ The value assigned to the node
--   -> BVar s Double
--     -- ^ The resulting (sum of) potential(s) (log domain!)
-- insideLogMemo graph potMap nodMap =
--   inside
--   where
--     inside = Memo.memo2 Memo.integral Memo.bool inside'
--     inside' v x =
--       hereLog `mulLog` downLog
--       where
--         hereLog = thereLog nodMap v x
--         downLog = productLog $ do
--           -- for each child
--           c <- incoming v graph
--           -- return the corresponding sum
--           return . sumLog $ do
--             -- for each corresponding arc value
--             y <- arcValues
--             -- such that the head of the value = x
--             guard $ hedVal y == x
--             -- return the resulting (exponential) potential
--             return $
--               arcPot potMap (c, v) y `mulLog`
--               inside c (depVal y)


-- | Inside pass (log domain)
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


-- -- | Inside pass (log domain)
-- insideLogMemo
--   :: (Reifies s W)
--   => Graph a b
--     -- ^ The underlying graph
--   -> M.Map Arc (BVar s (Vec8 Pot))
--     -- ^ The corresponding arc potential map (normal domain!)
--   -> M.Map G.Vertex (BVar s Double)
--     -- ^ The corresponding node potential map (normal domain!)
--   -> G.Vertex
--     -- ^ The node in question
--   -> Bool
--     -- ^ The value assigned to the node
--   -> BVar s Double
--     -- ^ The resulting (sum of) potential(s) (log domain!)
-- insideLogMemo graph potMap nodMap =
--   \v x -> fullMemo M.! (v, x)
--   where
--     fullMemo = List.foldl' inside M.empty ordered
--     ordered =
--       [ (v, x)
--       | v <- topoSort graph
--       , x <- [False, True] ]
--     inside memo (v, x) =
--       M.insert (v, x) (hereLog `mulLog` downLog) memo
--       where
--         hereLog = thereLog nodMap v x
--         downLog = productLog $ do
--           -- for each child
--           c <- incoming v graph
--           -- return the corresponding sum
--           return . sumLog $ do
--             -- for each corresponding arc value
--             y <- arcValues
--             -- such that the head of the value = x
--             guard $ hedVal y == x
--             -- return the resulting (exponential) potential
--             return $
--               arcPot potMap (c, v) y `mulLog`
--               (memo M.! (c, depVal y))
-- 
-- 
-- -- | Return the list of nodes sorted topologically (assumes that the input
-- -- graph is a tree). 
-- topoSort :: Graph a b -> [G.Vertex]
-- topoSort g =
--   reverse $ go (treeRoot g)
--   where
--     go v = v : concatMap go (Graph.incoming v g)


----------------------------------------------
-- Basic operations in log domain
----------------------------------------------


-- | Multiplication (log domain)
mulLog :: Floating a => a -> a -> a
mulLog = (+)
{-# INLINE mulLog #-}


-- -- | Sum (log domain)
-- sumLog :: Floating a => [a] -> a
-- sumLog = log . sum . map exp
-- {-# INLINE sumLog #-}


-- | Addition (log domain)
-- TODO: provide something more efficient?
addLog :: (Floating a, Ord a) => a -> a -> a
addLog x y = sumLog [x, y]
{-# INLINE addLog #-}


-- | Sum (log domain)
sumLog :: (Floating a, Ord a) => [a] -> a
sumLog xs =
  -- Code copied from the log-float library
  theMax + log theSum
  where
    theMax = maximum xs
    theSum = List.foldl' (\acc x -> acc + exp (x - theMax)) 0 xs
{-# INLINE sumLog #-}


-- | Product (log domain)
productLog :: Floating a => [a] -> a
productLog = sum
{-# INLINE productLog #-}
