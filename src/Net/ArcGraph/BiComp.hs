{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE MultiParamTypeClasses #-}
-- {-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
-- {-# LANGUAGE PatternSynonyms #-}
-- {-# LANGUAGE LambdaCase #-}
-- {-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TupleSections #-}

-------------------------
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=5 #-}
-------------------------


module Net.ArcGraph.BiComp
  ( BiComp (..)
  , Bias (..)
  , ArcBias (..)
  , PapyArcAff (..)
  , EnkelArcAff (..)
  -- , Aff (..)
  , HeadAff (..)
  , DepAff (..)
  , SymAff (..)
  -- , Pos (..)
  , HeadPosAff (..)
  , DepPosAff (..)
  , PapyPosAff (..)
  , EnkelPosAff (..)
  , Biaff (..)
  , UnordBiaff (..)
  , DirBiaff (..)
  , Holi (..)
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
import qualified GHC.TypeNats as Nats

import           Control.DeepSeq (NFData)
import           Control.Lens.At (ixAt)
import           Control.Monad (forM)

import           Lens.Micro.TH (makeLenses)

-- import           Data.Proxy (Proxy(..))
import qualified Data.Graph as G
import           Data.Binary (Binary)
import           Data.Maybe (mapMaybe)
import qualified Data.Set as S
import qualified Data.Map.Strict as M

import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, Reifies, W, BVar, (^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop (R, dot, (#))

import           Numeric.SGD.ParamSet (ParamSet)

import           Net.Basic hiding (scale)
import           Net.ArcGraph.Graph
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))

import           Debug.Trace (trace)


----------------------------------------------
-- Various Utils
----------------------------------------------


-- | Remove duplicates.
nub :: (Ord a) => [a] -> [a]
nub = S.toList . S.fromList


----------------------------------------------
-- Components
----------------------------------------------


-- | Biaffinity component
class Backprop comp => BiComp dim a b comp where
  runBiComp 
    :: (Reifies s W)
    => Graph (Node dim a) b
    -> Arc
    -> BVar s comp
    -> BVar s Double

instance (BiComp dim a b comp1, BiComp dim a b comp2)
  => BiComp dim a b (comp1 :& comp2) where
  runBiComp graph arc (comp1 :&& comp2) =
    runBiComp graph arc comp1 + runBiComp graph arc comp2


----------------------------------------------
----------------------------------------------


-- | Global bias
newtype Bias = Bias
  { _biasVal :: Double
  } deriving (Show, Generic)
    deriving newtype (Binary, NFData, ParamSet)
instance Backprop Bias
makeLenses ''Bias
instance New a b Bias where
  new xs ys = Bias <$> new xs ys
instance BiComp dim a b Bias where
  runBiComp _ _ bias = bias ^^. biasVal


----------------------------------------------
----------------------------------------------


-- | Arc label bias
newtype ArcBias b = ArcBias
  { _arcBiasMap :: M.Map b Double
  } deriving (Show, Generic)
    deriving newtype (Binary, NFData, ParamSet)

instance (Ord b) => Backprop (ArcBias b)
makeLenses ''ArcBias

instance (Ord b) => New a b (ArcBias b) where
  new xs ys = ArcBias <$> newMap ys xs ys


-- | Label affinity of the given arc
arcAff
  :: (Ord a, Show a, Reifies s W)
  => a -> BVar s (ArcBias a) -> BVar s Double
arcAff x aff = maybe err id $ do
  aff ^^. arcBiasMap ^^? ixAt x
  where
    err = trace
      ( "ArcGraph.arcAff: unknown arc label ("
      ++ show x
      ++ ")" ) 0

instance (Ord b, Show b) => BiComp dim a b (ArcBias b) where
  runBiComp graph (v, w) arcBias =
    let err = trace
          ( "ArcGraph.run: unknown arc ("
          ++ show (v, w)
          ++ ")" ) 0
     in maybe err id $ do
          arcLabel <- M.lookup (v, w) (arcLabelMap graph)
          return $ arcAff arcLabel arcBias


-- | Parent/grandparent arc label affinity
data PapyArcAff b = PapyArcAff
  { _papyArcAff :: ArcBias b
  , _papyArcDef :: Double
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''PapyArcAff
instance (Ord b) => New a b (PapyArcAff b) where
  new xs ys = PapyArcAff <$> new xs ys <*> pure 0
instance (Ord b, Show b) => BiComp dim a b (PapyArcAff b) where
  runBiComp graph (_, w) aff = check $ do
    x <- nub . mapMaybe (arcLabel.(w,)) $ outgoing w graph
    return $ arcAff x (aff ^^. papyArcAff)
    where
      arcLabel arc = M.lookup arc (arcLabelMap graph)
      check [] = aff ^^. papyArcDef
      check xs = sum xs


-- | Child/grandchild arc label affinity
data EnkelArcAff b = EnkelArcAff
  { _enkelnArcAff :: ArcBias b
  , _enkelnArcDef :: Double
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''EnkelArcAff
instance (Ord b) => New a b (EnkelArcAff b) where
  new xs ys = EnkelArcAff <$> new xs ys <*> pure 0
instance (Ord b, Show b) => BiComp dim a b (EnkelArcAff b) where
  runBiComp graph (v, _) aff = check $ do
    x <- nub . mapMaybe (arcLabel.(,v)) $ incoming v graph
    return $ arcAff x (aff ^^. enkelnArcAff)
    where
      arcLabel arc = M.lookup arc (arcLabelMap graph)
      check [] = aff ^^. enkelnArcDef
      check xs = sum xs


----------------------------------------------
----------------------------------------------


-- | Word affinity component
data Aff d h = Aff
  { _affN :: FFN
      d
      h -- Hidden layer size
      d -- Output
    -- ^ Potential FFN
  , _affV :: R d
    -- ^ Potential ,,feature vector''
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat dim, KnownNat h) => Backprop (Aff dim h)
makeLenses ''Aff

instance (KnownNat dim, KnownNat h) => New a b (Aff dim h) where
  new xs ys = Aff
    <$> new xs ys
    <*> new xs ys
    -- <*> vector d
--     where
--       d = proxyVal (Proxy :: Proxy dim)
--       h = proxyVal (Proxy :: Proxy h)
--       proxyVal = fromInteger . toInteger . natVal


-- | Affinity of the given node
nodeAff :: (KnownNat d, KnownNat h, Reifies s W)
        => Graph (Node d a) b -> G.Vertex -> BVar s (Aff d h) -> BVar s Double
nodeAff graph v aff =
  let nodeMap = fmap
        (BP.constVar . nodeEmb)
        (nodeLabelMap graph)
      hv = nodeMap M.! v
   in (aff ^^. affV) `dot`
        (FFN.run (aff ^^. affN) hv)


newtype HeadAff d h = HeadAff { _unHeadAff :: Aff d h }
  deriving (Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''HeadAff
instance (KnownNat dim, KnownNat h) => New a b (HeadAff dim h) where
  new xs ys = HeadAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim a b (HeadAff dim h) where
  runBiComp graph (_, w) aff = nodeAff graph w (aff ^^. unHeadAff)


newtype DepAff d h = DepAff { _unDepAff :: Aff d h }
  deriving (Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''DepAff
instance (KnownNat dim, KnownNat h) => New a b (DepAff dim h) where
  new xs ys = DepAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim a b (DepAff dim h) where
  runBiComp graph (v, _) aff = nodeAff graph v (aff ^^. unDepAff)


newtype SymAff d h = SymAff { _unSymAff :: Aff d h }
  deriving (Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''SymAff
instance (KnownNat dim, KnownNat h) => New a b (SymAff dim h) where
  new xs ys = SymAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim a b (SymAff dim h) where
  runBiComp graph (v, w) aff
    = nodeAff graph v (aff ^^. unSymAff)
    + nodeAff graph w (aff ^^. unSymAff)


-- instance (KnownNat dim, KnownNat h) => BiComp dim a b (Aff dim h) where
--   runBiComp graph (v, w) aff =
--      let nodeMap = fmap
--            (BP.constVar . nodeEmb)
--            (nodeLabelMap graph)
--          hv = nodeMap M.! v
--          hw = nodeMap M.! w
--          pot1 = (aff ^^. affV) `dot`
--            (FFN.run (aff ^^. affN) hv)
--          pot2 = (aff ^^. affV) `dot`
--            (FFN.run (aff ^^. affN) hw)
--       in pot1 + pot2


----------------------------------------------
----------------------------------------------


-- | POS affinity component
data Pos a = Pos
  { _posMap :: M.Map a Double
  } deriving (Show, Generic, Binary, NFData, ParamSet)

instance (Ord a) => Backprop (Pos a)
makeLenses ''Pos

instance (Ord a) => New a b (Pos a) where
  new xs ys = Pos <$> newMap xs xs ys


-- | POS affinity of the given node
posAff
  :: (Ord a, Show a, Reifies s W)
  => a -> BVar s (Pos a) -> BVar s Double
posAff pos aff = maybe err id $ do
  aff ^^. posMap ^^? ixAt pos
  where
    err = trace
      ( "ArcGraph.posAff: unknown POS ("
      ++ show pos
      ++ ")" ) 0


-- | POS affinity of the given node
nodePosAff
  :: (Ord a, Show a, Reifies s W)
  => Graph (Node d a) b -> G.Vertex
  -> BVar s (Pos a) -> BVar s Double
nodePosAff graph v aff = maybe err id $ do
  pos <- nodeLab <$> M.lookup v (nodeLabelMap graph)
  return (posAff pos aff)
  where
    err = trace
      ( "ArcGraph.nodePosAff: undefined node ("
      ++ show v
      ++ ")" ) 0


newtype HeadPosAff a = HeadPosAff { _unHeadPosAff :: Pos a }
  deriving (Show, Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''HeadPosAff
instance (Ord a) => New a b (HeadPosAff a) where
  new xs ys = HeadPosAff <$> new xs ys
instance (Ord a, Show a) => BiComp dim a b (HeadPosAff a) where
  runBiComp graph (_, w) aff = nodePosAff graph w (aff ^^. unHeadPosAff)


newtype DepPosAff a = DepPosAff { _unDepPosAff :: Pos a }
  deriving (Show, Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''DepPosAff
instance (Ord a) => New a b (DepPosAff a) where
  new xs ys = DepPosAff <$> new xs ys
instance (Ord a, Show a) => BiComp dim a b (DepPosAff a) where
  runBiComp graph (v, _) aff = nodePosAff graph v (aff ^^. unDepPosAff)


data PapyPosAff a = PapyPosAff
  { _papyPosAff :: Pos a 
  , _papyPosDef :: Double
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''PapyPosAff
instance (Ord a) => New a b (PapyPosAff a) where
  new xs ys = PapyPosAff <$> new xs ys <*> pure 0
instance (Ord a, Show a) => BiComp dim a b (PapyPosAff a) where
  runBiComp graph (_, w) aff = check $ do
    pos <- nub . mapMaybe (posTag graph) $ outgoing w graph
    return $ posAff pos (aff ^^. papyPosAff)
    where
      check [] = aff ^^. papyPosDef
      check xs = sum xs


data EnkelPosAff a = EnkelPosAff
  { _enkelPosAff :: Pos a 
  , _enkelPosDef :: Double
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''EnkelPosAff
instance (Ord a) => New a b (EnkelPosAff a) where
  new xs ys = EnkelPosAff <$> new xs ys <*> pure 0
instance (Ord a, Show a) => BiComp dim a b (EnkelPosAff a) where
  runBiComp graph (v, _) aff = check $ do
    pos <- nub . mapMaybe (posTag graph) $ incoming v graph
    return $ posAff pos (aff ^^. enkelPosAff)
    where
      check [] = aff ^^. enkelPosDef
      check xs = sum xs


----------------------------------------------
----------------------------------------------


-- | Biaffinity component
data Biaff d h = Biaff
  { _biaffN :: FFN
      (d Nats.+ d)
      -- TODO: make hidden layer larger?  The example with the grenade library
      -- suggests this.
      h -- Hidden layer size
      d -- Output
    -- ^ Potential FFN
  , _biaffV :: R d
    -- ^ Potential ,,feature vector''
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat dim, KnownNat h) => Backprop (Biaff dim h)
makeLenses ''Biaff

instance (KnownNat dim, KnownNat h) => New a b (Biaff dim h) where
  new xs ys = Biaff
--     <$> FFN.new (d*2) h d
--     <*> vector d
    <$> new xs ys
    <*> new xs ys
--     where
--       d = proxyVal (Proxy :: Proxy dim)
--       h = proxyVal (Proxy :: Proxy h)
--       proxyVal = fromInteger . toInteger . natVal

instance (KnownNat dim, KnownNat h) => BiComp dim a b (Biaff dim h) where
  runBiComp graph (v, w) bia =
     let nodeMap = fmap
           (BP.constVar . nodeEmb)
           (nodeLabelMap graph)
         hv = nodeMap M.! v
         hw = nodeMap M.! w
      in (bia ^^. biaffV) `dot`
           (FFN.run (bia ^^. biaffN) (hv # hw))


----------------------------------------------
----------------------------------------------


-- | Unordered biaffinity component
data UnordBiaff d h = UnordBiaff
  { _unordBiaffN :: FFN
      (d Nats.+ d)
      h -- Hidden layer size
      d -- Output
    -- ^ Potential FFN
  , _unordBiaffV :: R d
    -- ^ Potential ,,feature vector''
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat dim, KnownNat h) => Backprop (UnordBiaff dim h)
makeLenses ''UnordBiaff

instance (KnownNat dim, KnownNat h) => New a b (UnordBiaff dim h) where
  new xs ys = UnordBiaff
--     <$> FFN.new (d*2) h d
--     <*> vector d
    <$> new xs ys
    <*> new xs ys
--     where
--       d = proxyVal (Proxy :: Proxy dim)
--       h = proxyVal (Proxy :: Proxy h)
--       proxyVal = fromInteger . toInteger . natVal

instance (KnownNat dim, KnownNat h) => BiComp dim a b (UnordBiaff dim h) where
  runBiComp graph (v, w) unordBia =
     let vLex = nodeLex <$> M.lookup v (nodeLabelMap graph)
         wLex = nodeLex <$> M.lookup w (nodeLabelMap graph)
         embMap = fmap
           (BP.constVar . nodeEmb)
           (nodeLabelMap graph)
         emb = (embMap M.!)
         (hv, hw) =
           if vLex <= wLex
              then (emb v, emb w)
              else (emb w, emb v)
      in (unordBia ^^. unordBiaffV) `dot`
           (FFN.run (unordBia ^^. unordBiaffN) (hv # hw))


----------------------------------------------
----------------------------------------------


-- | A version of `Biaff` in which the label between two nodes determines who
-- is the ,,semantic'' parent.
data DirBiaff dim h b = DirBiaff
  { _dirBiaMap :: M.Map b Double
  , _dirBiaff :: Biaff dim h
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat dim, KnownNat h, Ord b) => Backprop (DirBiaff dim h b)
makeLenses ''DirBiaff

instance (KnownNat dim, KnownNat h, Ord b) => New a b (DirBiaff dim h b) where
  new nodeLabelSet arcLabelSet = DirBiaff
    -- TODO: could be simplified...
    <$> mkMap arcLabelSet (pure 0)
    <*> new nodeLabelSet arcLabelSet
    where
      mkMap keySet mkVal = fmap M.fromList .
        forM (S.toList keySet) $ \key -> do
          (key,) <$> mkVal

instance (KnownNat dim, KnownNat h, Ord b, Show b)
  => BiComp dim a b (DirBiaff dim h b) where
  runBiComp graph (v, w) dirBia =
    let nodeMap = fmap
          (BP.constVar . nodeEmb)
          (nodeLabelMap graph)
        hv0 = nodeMap M.! v
        hw0 = nodeMap M.! w
        err = trace
          ( "ArcGraph.DirBiaff: unknown arc label ("
          ++ show (M.lookup (v, w) (arcLabelMap graph))
          ++ ")" ) 0
        p = maybe err logistic $ do
              arcLabel <- M.lookup (v, w) (arcLabelMap graph)
              dirBia ^^. dirBiaMap ^^? ixAt arcLabel
        q = 1 - p
        scale x = LBP.vmap (*x)
        hv = scale p hv0 + scale q hw0
        hw = scale q hv0 + scale p hw0
        bia = dirBia ^^. dirBiaff
     in (bia ^^. biaffV) `dot`
          (FFN.run (bia ^^. biaffN) (hv # hw))


----------------------------------------------
----------------------------------------------


-- -- | A version of `Biaff` in which the label between two nodes determines who
-- -- is the ,,semantic'' parent.
-- data DirBiaff dim h b = DirBiaff
--   { _dirBiaMap :: M.Map b Double
--   , _dirBiaff :: Biaff dim h
--   } deriving (Generic, Binary, NFData, ParamSet)
-- 
-- instance (KnownNat dim, KnownNat h, Ord b) => Backprop (DirBiaff dim h b)
-- makeLenses ''DirBiaff
-- 
-- instance (KnownNat dim, KnownNat h, Ord b) => New a b (DirBiaff dim h b) where
--   new nodeLabelSet arcLabelSet = DirBiaff
--     -- TODO: could be simplified...
--     <$> mkMap arcLabelSet (pure 0)
--     <*> new nodeLabelSet arcLabelSet
--     where
--       mkMap keySet mkVal = fmap M.fromList .
--         forM (S.toList keySet) $ \key -> do
--           (key,) <$> mkVal
-- 
-- instance (KnownNat dim, KnownNat h, Ord b, Show b)
--   => BiComp dim a b (DirBiaff dim h b) where
--   runBiComp graph (v, w) dirBia =
--     let nodeMap = fmap
--           (BP.constVar . nodeEmb)
--           (nodeLabelMap graph)
--         hv = nodeMap M.! v
--         hw = nodeMap M.! w
--         err = trace
--           ( "ArcGraph.DirBiaff: unknown arc label ("
--           ++ show (M.lookup (v, w) (arcLabelMap graph))
--           ++ ")" ) 1
--         -- logist x = 1.0 / (1.0 + exp (-100*x))
--         logist = logistic
--         p = maybe err logist $ do
--               arcLabel <- M.lookup (v, w) (arcLabelMap graph)
--               dirBia ^^. dirBiaMap ^^? ixAt arcLabel
--         q = 1 - p
--         bia = dirBia ^^. dirBiaff
--         pot1 = (bia ^^. biaffV) `dot`
--                   (FFN.run (bia ^^. biaffN) (hv # hw))
--         pot2 = (bia ^^. biaffV) `dot`
--                   (FFN.run (bia ^^. biaffN) (hw # hv))
--      in if p > 0.99
--         then p*pot1
--         else if q > 0.99
--         then q*pot2
--         else p*pot1 + q*pot2


----------------------------------------------
----------------------------------------------


-- | Holistic compoment -- (almost) all in one
--
--   * @d1@ -- word embedding dimension
--   * @d2@ -- label embedding dimension
--   * @h@  -- hidden dimension
--   * @o@  -- pre-output dimension
--
data Holi d1 d2 a b h o = Holi
  { _holN :: FFN
      (d1 Nats.+ d1 Nats.+ d2 Nats.+ d2 Nats.+ d2)
      -- (d1 Nats.+ d1 Nats.+ d2 Nats.+ d2)
        -- two words + two POS + dep label
      h
        -- hidden layer size
      o
        -- output layer size
  , _holV :: R o
      -- ^ To get the ultimate potential value via dot product
  , _holHeadPosMap :: M.Map a (R d2)
      -- ^ Head POS repr
  , _holDepPosMap :: M.Map a (R d2)
      -- ^ Dependent POS repr
  , _holArcMap :: M.Map b (R d2)
      -- ^ Arc label repr
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat d1, KnownNat d2, Ord a, Ord b, KnownNat h, KnownNat o)
  => Backprop (Holi d1 d2 a b h o)
makeLenses ''Holi

instance (KnownNat d1, KnownNat d2, Ord a, Ord b, KnownNat h, KnownNat o)
  => New a b (Holi d1 d2 a b h o) where
  new xs ys = Holi
    <$> new xs ys
    <*> new xs ys
    <*> newMap xs xs ys
    <*> newMap xs xs ys
    <*> newMap ys xs ys

instance
  ( KnownNat d1, KnownNat d2, KnownNat h, KnownNat o
  , Show a, Ord a, Show b, Ord b )
  => BiComp d1 a b (Holi d1 d2 a b h o) where
    runBiComp graph (v, w) holi =
      let nodeMap = nodeLabelMap graph
          wordMap = fmap (BP.constVar . nodeEmb) nodeMap
          hv = wordMap M.! v
          hw = wordMap M.! w
          depPos = depPosRepr . nodeLab $ nodeMap M.! v
          headPos = headPosRepr . nodeLab $ nodeMap M.! w
          arc = arcRepr $ arcLabelMap graph M.! (v, w)
       in (holi ^^. holV) `dot`
            (FFN.run (holi ^^. holN) (hv # hw # depPos # headPos # arc))
      where
        headPosRepr pos = maybe err id $ do
          holi ^^. holHeadPosMap ^^? ixAt pos
          where
            err = trace
              ( "ArcGraph.Holi: unknown POS ("
              ++ show pos
              ++ ")" ) 0
        depPosRepr pos = maybe err id $ do
          holi ^^. holDepPosMap ^^? ixAt pos
          where
            err = trace
              ( "ArcGraph.Holi: unknown POS ("
              ++ show pos
              ++ ")" ) 0
        arcRepr dep = maybe err id $ do
          holi ^^. holArcMap ^^? ixAt dep
          where
            err = trace
              ( "ArcGraph.Holi: unknown arc label ("
              ++ show dep
              ++ ")" ) 0
