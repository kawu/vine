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
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveFoldable #-}
-- {-# LANGUAGE DeriveTraversable #-}

-------------------------
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE ScopedTypeVariables #-}

{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=5 #-}
-------------------------


module Net.Graph2.BiComp
  ( Pot
  , Prob
  , Vec(..)
  , Vec8
  , softmaxVec
  
  , BiComp (..)
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
  , BiAff (..)
  , BiAffExt (..)
  , BiAffMix (..)
  , UnordBiAff (..)
--   , DirBiaff (..)
--   , Holi (..)


  -- * Vec3 <-> Vec8 conversion
  , Out(..)
  , enumerate
  , mask
  , squash
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
import qualified GHC.TypeNats as Nats

import           Control.DeepSeq (NFData)
import           Control.Lens.At (ixAt, ix)
import qualified Control.Lens.At as At
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
import qualified Numeric.LinearAlgebra.Static as LA
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop (R, L, dot, (#), (#>))

import qualified Test.SmallCheck.Series as SC

import           Numeric.SGD.ParamSet (ParamSet)

import           Graph
import           Net.Util hiding (scale)
import           Net.New
import           Net.Pair
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
-- Vec8 <-> Vec3 conversion
----------------------------------------------


-- -- | V8 -> V3 squashing
-- --
-- -- The `squash` function is a backpropagation-enabled version of @decode@ from
-- -- `Net.Graph2`.  The result is a vector of three probability values:
-- --
-- --   * Probability of the arc being a MWE
-- --   * Probability of the head being a MWE
-- --   * Probability of the dependent being a MWE
-- --
-- squash
--   :: (Reifies s W)
--   => BVar s (Vec 8 Prob)
--   -> BVar s (R 3)
-- squash v8 = undefined


-- | Output structure in which a value of type @a@ is assigned to an arc and
-- the two nodes it connects.
data Out a = Out
  { arcVal :: a
    -- ^ Value assigned to the arc
  , hedVal :: a
    -- ^ Value assigned to the head
  , depVal :: a
    -- ^ Value assigned to the dependent
  } deriving (Generic, Show, Eq, Ord, Functor, Foldable) --, Traversable)

-- Allows to use SmallCheck to test (decode . encode) == id.
instance (SC.Serial m a) => SC.Serial m (Out a)


-- | Enumerate the possible arc/node labelings in order consistent with the
-- encoding/decoding format.
enumerate :: [Out Bool]
enumerate = do
  b1 <- [False, True]
  b2 <- [False, True]
  b3 <- [False, True]
  return $ Out b1 b2 b3


-- | A mask vector which allows to easily obtain (with dot product) the
-- potential of a given `Out` labeling.
--
-- TODO: the mask could be perhaps calculated using bit-level operattions?
--
mask :: Out Bool -> R 8
mask (Out False False False) = vec18
mask (Out False False True)  = vec28
mask (Out False True  False) = vec38
mask (Out False True  True)  = vec48
mask (Out True  False False) = vec58
mask (Out True  False True)  = vec68
mask (Out True  True  False) = vec78
mask (Out True  True  True)  = vec88


-- | Hard-coded masks
vec18, vec28, vec38, vec48, vec58, vec68, vec78, vec88 :: R 8
vec18 = LA.vector [1, 0, 0, 0, 0, 0, 0, 0]
vec28 = LA.vector [0, 1, 0, 0, 0, 0, 0, 0]
vec38 = LA.vector [0, 0, 1, 0, 0, 0, 0, 0]
vec48 = LA.vector [0, 0, 0, 1, 0, 0, 0, 0]
vec58 = LA.vector [0, 0, 0, 0, 1, 0, 0, 0]
vec68 = LA.vector [0, 0, 0, 0, 0, 1, 0, 0]
vec78 = LA.vector [0, 0, 0, 0, 0, 0, 1, 0]
vec88 = LA.vector [0, 0, 0, 0, 0, 0, 0, 1]


-- -- | Cross entropy between the true and the artificial distributions
-- crossEntropy
--   :: forall s. (Reifies s W)
--   => BVar s (Vec 8 Prob)
--     -- ^ Target ,,true'' probability distribution
--   -> BVar s (Vec 8 Prob)
--     -- ^ Output ,,artificial'' distribution
--   -> BVar s Double
-- crossEntropy p0 q0 =
--   undefined


-- | V8 -> V3 squashing
--
-- The `squash` function is a backpropagation-enabled version of @decode@ from
-- `Net.Graph2`.  The result is a vector of three probability values:
--
--   * Probability of the arc being a MWE
--   * Probability of the head being a MWE
--   * Probability of the dependent being a MWE
--
squash :: forall s. (Reifies s W) => BVar s (Vec 8 Prob) -> Out (BVar s Double)
squash v8_vec = Out
  { arcVal = BP.auto mask1 `dot` v8
  , hedVal = BP.auto mask2 `dot` v8
  , depVal = BP.auto mask3 `dot` v8
  } 
  where
    v8 = BP.coerceVar v8_vec :: BVar s (R 8)


-- | V3 -> V8 expansion
--
-- TODO: Make some kind of link between `epxand` and @encode@ from
-- `Net.Graph2`.
--
expand
  :: (Reifies s W)
  => BVar s (Vec 3 Pot)
  -> BVar s (Vec 8 Pot)
expand v3 = BP.coerceVar $ expand' (BP.coerceVar v3)


-- | Combine the independent with the joint potential vector (lower-level
-- function).
expand' :: (Reifies s W) => BVar s (R 3) -> BVar s (R 8)
expand' v3 
  = LBP.vmap (*x1) (BP.auto mask1)
  + LBP.vmap (*x2) (BP.auto mask2)
  + LBP.vmap (*x3) (BP.auto mask3)
  where
    v3' = LBP.extractV v3
    x1 = v3' `at` 0
    x2 = v3' `at` 1
    x3 = v3' `at` 2
{-# INLINE expand' #-}


-- | Expansion masks
mask0, mask1, mask2, mask3 :: R 8
mask0 = LA.vector [0, 1, 1, 1, 1, 1, 1, 1]
mask1 = LA.vector [0, 0, 0, 0, 1, 1, 1, 1]
mask2 = LA.vector [0, 0, 1, 1, 0, 0, 1, 1]
mask3 = LA.vector [0, 1, 0, 1, 0, 1, 0, 1]
{-# NOINLINE mask0 #-}
{-# NOINLINE mask1 #-}
{-# NOINLINE mask2 #-}
{-# NOINLINE mask3 #-}


at
  :: ( Num (At.IxValue b), Reifies s W, Backprop b
     , Backprop (At.IxValue b), At.Ixed b
     )
  => BVar s b
  -> At.Index b
  -> BVar s (At.IxValue b)
at v k = maybe 0 id $ v ^^? ix k
{-# INLINE at #-}


----------------------------------------------
-- Components
----------------------------------------------


-- | Potential/probability annotation
data Pot
data Prob


-- | Output potential is a vector of length 2^3 (the number of possible
-- combinations of values of the:
--
--   * label of the head
--   * label of the dependent
--   * label of the arc
--
-- We additionally trace the size of the vector @n@.  It's actually just for
-- convenience, to not to have to tell in other places in the code that the
-- size of `unVec` is actually @8@.
--
newtype Vec n p = Vec { unVec :: R n }
  deriving (Show, Generic)
  deriving newtype (Binary, NFData, ParamSet, Num, Backprop)

instance (KnownNat n) => New a b (Vec n p) where
  new xs ys = Vec <$> new xs ys


-- | Softmax over a vector of potentials
softmaxVec 
  :: forall n s. (KnownNat n, Reifies s W)
  => BVar s (Vec n Pot) -> BVar s (Vec n Prob)
softmaxVec
  = BP.coerceVar
  . (softmax :: Reifies s W => BVar s (R n) -> BVar s (R n))
  . BP.coerceVar


-- | Type synonym to @Vec 8 p@.
type Vec8 p = Vec 8 p


-- | Biaffinity component
class Backprop comp => BiComp dim a b comp where
  runBiComp 
    :: (Reifies s W)
    => Graph (Node dim a) b
    -> Arc
    -> BVar s comp
    -> BVar s (Vec8 Pot)

instance (BiComp dim a b comp1, BiComp dim a b comp2)
  => BiComp dim a b (comp1 :& comp2) where
  runBiComp graph arc (comp1 :&& comp2) =
    runBiComp graph arc comp1 + runBiComp graph arc comp2


----------------------------------------------
----------------------------------------------


-- | Global bias
newtype Bias = Bias
  { _biasVal :: Vec8 Pot
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
  { _arcBiasMap :: M.Map b (Vec8 Pot)
  } deriving (Show, Generic)
    deriving newtype (Binary, NFData, ParamSet)

instance (Ord b) => Backprop (ArcBias b)
makeLenses ''ArcBias

instance (Ord b) => New a b (ArcBias b) where
  new xs ys = ArcBias <$> newMap ys xs ys


-- | Label affinity of the given arc
arcAff
  :: (Ord a, Show a, Reifies s W)
  => a -> BVar s (ArcBias a) -> BVar s (Vec8 Pot)
arcAff x aff = maybe err id $ do
  aff ^^. arcBiasMap ^^? ixAt x
  where
    err = trace
      ( "Graph.arcAff: unknown arc label ("
      ++ show x
      ++ ")" ) 0

instance (Ord b, Show b) => BiComp dim a b (ArcBias b) where
  runBiComp graph (v, w) arcBias =
    let err = trace
          ( "Graph.run: unknown arc ("
          ++ show (v, w)
          ++ ")" ) 0
     in maybe err id $ do
          arcLabel <- M.lookup (v, w) (arcLabelMap graph)
          return $ arcAff arcLabel arcBias


-- | Parent/grandparent arc label affinity
data PapyArcAff b = PapyArcAff
  { _papyArcAff :: ArcBias b
  , _papyArcDef :: Vec8 Pot
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
  , _enkelnArcDef :: Vec8 Pot
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
data WordAff d h = WordAff
  { _wordAffN :: FFN d h 8
  } deriving (Generic, Binary, NFData, ParamSet)

instance (KnownNat dim, KnownNat h) => Backprop (WordAff dim h)
makeLenses ''WordAff

instance (KnownNat dim, KnownNat h) => New a b (WordAff dim h) where
  new xs ys = WordAff <$> new xs ys


-- | Affinity of the given node
nodeAff :: (KnownNat d, KnownNat h, Reifies s W)
        => Graph (Node d a) b
        -> G.Vertex
        -> BVar s (WordAff d h)
        -> BVar s (Vec8 Pot)
nodeAff graph v aff =
  let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
      hv = wordRepr v
   in BP.coerceVar $ FFN.run (aff ^^. wordAffN) hv


newtype HeadAff d h = HeadAff { _unHeadAff :: WordAff d h }
  deriving (Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''HeadAff
instance (KnownNat dim, KnownNat h) => New a b (HeadAff dim h) where
  new xs ys = HeadAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim a b (HeadAff dim h) where
  runBiComp graph (_, w) aff = nodeAff graph w (aff ^^. unHeadAff)


newtype DepAff d h = DepAff { _unDepAff :: WordAff d h }
  deriving (Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''DepAff
instance (KnownNat dim, KnownNat h) => New a b (DepAff dim h) where
  new xs ys = DepAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim a b (DepAff dim h) where
  runBiComp graph (v, _) aff = nodeAff graph v (aff ^^. unDepAff)


newtype SymAff d h = SymAff { _unSymAff :: WordAff d h }
  deriving (Generic)
  deriving newtype (Binary, NFData, ParamSet, Backprop)
makeLenses ''SymAff
instance (KnownNat dim, KnownNat h) => New a b (SymAff dim h) where
  new xs ys = SymAff <$> new xs ys
instance (KnownNat dim, KnownNat h) => BiComp dim a b (SymAff dim h) where
  runBiComp graph (v, w) aff
    = nodeAff graph v (aff ^^. unSymAff)
    + nodeAff graph w (aff ^^. unSymAff)


----------------------------------------------
----------------------------------------------


-- | POS affinity component
data Pos a = Pos
  { _posMap :: M.Map a (Vec8 Pot)
  } deriving (Show, Generic, Binary, NFData, ParamSet)

instance (Ord a) => Backprop (Pos a)
makeLenses ''Pos

instance (Ord a) => New a b (Pos a) where
  new xs ys = Pos <$> newMap xs xs ys


-- | POS affinity of the given node
posAff
  :: (Ord a, Show a, Reifies s W)
  => a -> BVar s (Pos a) -> BVar s (Vec8 Pot)
posAff pos aff = maybe err id $ do
  aff ^^. posMap ^^? ixAt pos
  where
    err = trace
      ( "Graph.posAff: unknown POS ("
      ++ show pos
      ++ ")" ) 0


-- | POS affinity of the given node
nodePosAff
  :: (Ord a, Show a, Reifies s W)
  => Graph (Node d a) b -> G.Vertex
  -> BVar s (Pos a) -> BVar s (Vec8 Pot)
nodePosAff graph v aff = maybe err id $ do
  pos <- nodeLab <$> M.lookup v (nodeLabelMap graph)
  return (posAff pos aff)
  where
    err = trace
      ( "Graph.nodePosAff: undefined node ("
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
  , _papyPosDef :: Vec8 Pot
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''PapyPosAff
instance (Ord a) => New a b (PapyPosAff a) where
  new xs ys = PapyPosAff <$> new xs ys <*> pure 0
instance (Ord a, Show a) => BiComp dim a b (PapyPosAff a) where
  runBiComp graph (_, w) aff = check $ do
    pos <- nub . mapMaybe (nodeLabAt graph) $ outgoing w graph
    return $ posAff pos (aff ^^. papyPosAff)
    where
      check [] = aff ^^. papyPosDef
      check xs = sum xs


data EnkelPosAff a = EnkelPosAff
  { _enkelPosAff :: Pos a 
  , _enkelPosDef :: Vec8 Pot
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)
makeLenses ''EnkelPosAff
instance (Ord a) => New a b (EnkelPosAff a) where
  new xs ys = EnkelPosAff <$> new xs ys <*> pure 0
instance (Ord a, Show a) => BiComp dim a b (EnkelPosAff a) where
  runBiComp graph (v, _) aff = check $ do
    pos <- nub . mapMaybe (nodeLabAt graph) $ incoming v graph
    return $ posAff pos (aff ^^. enkelPosAff)
    where
      check [] = aff ^^. enkelPosDef
      check xs = sum xs


----------------------------------------------
----------------------------------------------


-- | Biaffinity component
data BiAff d h = BiAff
  { _biAffN :: FFN (d Nats.+ d) h 8
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''BiAff

instance (KnownNat dim, KnownNat h) => New a b (BiAff dim h) where
  new xs ys = BiAff <$> new xs ys

instance (KnownNat dim, KnownNat h) => BiComp dim a b (BiAff dim h) where
  runBiComp graph (v, w) bi =
    let wordRepr = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        hv = wordRepr v
        hw = wordRepr w
     in BP.coerceVar $ FFN.run (bi ^^. biAffN) (hv # hw)


----------------------------------------------
----------------------------------------------


-- | Biaffinity component extended with info about POS and DEP labels.
--
--   * @d@ -- word embedding dimension
--   * @l@ -- label (POS, DEP) embedding dimension
--   * @h@ -- hidden dimension
--
data BiAffExt d l a b h = BiAffExt
  { _biAffExtN :: FFN (d Nats.+ d Nats.+ l Nats.+ l Nats.+ l) h 8
    -- ^ The actual bi-affinity net
  , _biAffHeadPosMap :: M.Map a (R l)
    -- ^ Head POS repr
  , _biAffDepPosMap :: M.Map a (R l)
    -- ^ Dependent POS repr
  , _biAffArcMap :: M.Map b (R l)
    -- ^ Arc label repr
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''BiAffExt

instance (KnownNat d, KnownNat h, KnownNat l, Ord a, Ord b)
  => New a b (BiAffExt d l a b h) where
  new xs ys = BiAffExt
    <$> new xs ys
    <*> newMap xs xs ys
    <*> newMap xs xs ys
    <*> newMap ys xs ys

instance (KnownNat d, KnownNat h, KnownNat l, Ord a, Ord b, Show a, Show b)
  => BiComp d a b (BiAffExt d l a b h) where
  runBiComp graph (v, w) bi =
    let nodeMap = nodeLabelMap graph
        emb = BP.constVar . nodeEmb . (nodeMap M.!)
        depPos = depPosRepr . nodeLab $ nodeMap M.! v
        headPos = headPosRepr . nodeLab $ nodeMap M.! w
        arc = arcRepr $ arcLabelMap graph M.! (v, w)
        hv = emb v
        hw = emb w
     in BP.coerceVar $ FFN.run (bi ^^. biAffExtN) 
          (hv # hw # depPos # headPos # arc)
    where
      headPosRepr pos = maybe err id $ do
        bi ^^. biAffHeadPosMap ^^? ixAt pos
        where
          err = trace
            ( "Graph2.BiComp: unknown POS ("
            ++ show pos
            ++ ")" ) 0
      depPosRepr pos = maybe err id $ do
        bi ^^. biAffDepPosMap ^^? ixAt pos
        where
          err = trace
            ( "Graph2.BiComp: unknown POS ("
            ++ show pos
            ++ ")" ) 0
      arcRepr dep = maybe err id $ do
        bi ^^. biAffArcMap ^^? ixAt dep
        where
          err = trace
            ( "Graph2.BiComp: unknown arc label ("
            ++ show dep
            ++ ")" ) 0


----------------------------------------------
----------------------------------------------


-- | A version of `BiAffExt` where the decision about the labling of the arc
-- and its end nodes is taken partially independently.
--
--   * @d@ -- word embedding dimension
--   * @l@ -- label (POS, DEP) embedding dimension
--   * @h@ -- hidden dimension
--
data BiAffMix d l a b h = BiAffMix
  { _biAffMixL1 :: {-# UNPACK #-} !(L h (d Nats.+ d Nats.+ l Nats.+ l Nats.+ l))
    -- ^ First layer transforming the input vectors into the hidden
    -- representation
  , _biAffMixB1 :: {-# UNPACK #-} !(R h)
    -- ^ The bias corresponding to the first layer
  , _biAffMixL2_8 :: {-# UNPACK #-} !(L 8 h)
    -- ^ Second layer with joint potential attribution
  , _biAffMixB2_8 :: {-# UNPACK #-} !(R 8)
    -- ^ The bias corresponding to the second (joint) layer
  , _biAffMixL2_3 :: {-# UNPACK #-} !(L 3 h)
    -- ^ Second layer with independent potential attribution
  , _biAffMixB2_3 :: {-# UNPACK #-} !(R 3)
    -- ^ The bias corresponding to the second (independent) layer
  , _biAffMixHeadPosMap :: M.Map a (R l)
    -- ^ Head POS repr
  , _biAffMixDepPosMap :: M.Map a (R l)
    -- ^ Dependent POS repr
  , _biAffMixArcMap :: M.Map b (R l)
    -- ^ Arc label repr
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''BiAffMix

instance (KnownNat d, KnownNat h, KnownNat l, Ord a, Ord b)
  => New a b (BiAffMix d l a b h) where
  new xs ys = BiAffMix
    <$> new xs ys
    <*> new xs ys
    <*> new xs ys
    <*> new xs ys
    <*> new xs ys
    <*> new xs ys
    <*> newMap xs xs ys
    <*> newMap xs xs ys
    <*> newMap ys xs ys

instance (KnownNat d, KnownNat h, KnownNat l, Ord a, Ord b, Show a, Show b)
  => BiComp d a b (BiAffMix d l a b h) where

  runBiComp graph (v, w) bi =
    
    -- helper functions
    let nodeMap = nodeLabelMap graph
        emb = BP.constVar . nodeEmb . (nodeMap M.!)
        depPos = depPosRepr . nodeLab $ nodeMap M.! v
        headPos = headPosRepr . nodeLab $ nodeMap M.! w
        arc = arcRepr $ arcLabelMap graph M.! (v, w)
        hv = emb v
        hw = emb w

        -- input layer
        x = hv # hw # depPos # headPos # arc
        -- second layer
        y = leakyRelu $ (bi ^^. biAffMixL1) #> x + (bi ^^. biAffMixB1)
        -- joint output
        z8 = BP.coerceVar $ 
          (bi ^^. biAffMixL2_8) #> y + (bi ^^. biAffMixB2_8)
        -- independent output
        z3 = BP.coerceVar $
          (bi ^^. biAffMixL2_3) #> y + (bi ^^. biAffMixB2_3)

     -- combine the two outputs to get the result
     -- (with the first element zeroed out)
     in  BP.coerceVar (BP.auto mask0) * inject z3 z8

    where

      headPosRepr pos = maybe err id $ do
        bi ^^. biAffMixHeadPosMap ^^? ixAt pos
        where
          err = trace
            ( "Graph2.BiComp: unknown POS ("
            ++ show pos
            ++ ")" ) 0

      depPosRepr pos = maybe err id $ do
        bi ^^. biAffMixDepPosMap ^^? ixAt pos
        where
          err = trace
            ( "Graph2.BiComp: unknown POS ("
            ++ show pos
            ++ ")" ) 0

      arcRepr dep = maybe err id $ do
        bi ^^. biAffMixArcMap ^^? ixAt dep
        where
          err = trace
            ( "Graph2.BiComp: unknown arc label ("
            ++ show dep
            ++ ")" ) 0


-- | Combine the independent with the joint potential vector (a type-safe
-- wrapper over inject').
inject
  :: (Reifies s W)
  => BVar s (Vec 3 Pot)
  -> BVar s (Vec 8 Pot)
  -> BVar s (Vec 8 Pot)
inject v3 v8 = expand v3 + v8
{-# INLINE inject #-}


----------------------------------------------
----------------------------------------------


-- | Unordered bi-affinity component.
data UnordBiAff d h = UnordBiAff
  { _unordBiAffN :: FFN (d Nats.+ d) h 1
  } deriving (Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''UnordBiAff

instance (KnownNat d, KnownNat h) => New a b (UnordBiAff d h) where
  new xs ys = UnordBiAff <$> new xs ys

instance (KnownNat d, KnownNat h) => BiComp d a b (UnordBiAff d h) where
  runBiComp graph (v, w) bi =
    let lex x = nodeLex <$> M.lookup x (nodeLabelMap graph)
        emb = BP.constVar . nodeEmb . (nodeLabelMap graph M.!)
        (hv, hw) =
          if lex v <= lex w
             then (emb v, emb w)
             else (emb w, emb v)
     in BP.coerceVar $ FFN.run (bi ^^. unordBiAffN) (hv # hw)


----------------------------------------------
----------------------------------------------


-- -- | Unordered biaffinity component
-- data UnordBiaff d h = UnordBiaff
--   { _unordBiaffN :: FFN
--       (d Nats.+ d)
--       h -- Hidden layer size
--       d -- Output
--     -- ^ Potential FFN
--   , _unordBiaffV :: R d
--     -- ^ Potential ,,feature vector''
--   } deriving (Generic, Binary, NFData, ParamSet)
-- 
-- instance (KnownNat dim, KnownNat h) => Backprop (UnordBiaff dim h)
-- makeLenses ''UnordBiaff
-- 
-- instance (KnownNat dim, KnownNat h) => New a b (UnordBiaff dim h) where
--   new xs ys = UnordBiaff
-- --     <$> FFN.new (d*2) h d
-- --     <*> vector d
--     <$> new xs ys
--     <*> new xs ys
-- --     where
-- --       d = proxyVal (Proxy :: Proxy dim)
-- --       h = proxyVal (Proxy :: Proxy h)
-- --       proxyVal = fromInteger . toInteger . natVal
-- 
-- instance (KnownNat dim, KnownNat h) => BiComp dim a b (UnordBiaff dim h) where
--   runBiComp graph (v, w) unordBia =
--      let vLex = nodeLex <$> M.lookup v (nodeLabelMap graph)
--          wLex = nodeLex <$> M.lookup w (nodeLabelMap graph)
--          embMap = fmap
--            (BP.constVar . nodeEmb)
--            (nodeLabelMap graph)
--          emb = (embMap M.!)
--          (hv, hw) =
--            if vLex <= wLex
--               then (emb v, emb w)
--               else (emb w, emb v)
--       in (unordBia ^^. unordBiaffV) `dot`
--            (FFN.run (unordBia ^^. unordBiaffN) (hv # hw))
-- 
-- 
-- ----------------------------------------------
-- ----------------------------------------------
-- 
-- 
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
--         hv0 = nodeMap M.! v
--         hw0 = nodeMap M.! w
--         err = trace
--           ( "Graph.DirBiaff: unknown arc label ("
--           ++ show (M.lookup (v, w) (arcLabelMap graph))
--           ++ ")" ) 0
--         p = maybe err logistic $ do
--               arcLabel <- M.lookup (v, w) (arcLabelMap graph)
--               dirBia ^^. dirBiaMap ^^? ixAt arcLabel
--         q = 1 - p
--         scale x = LBP.vmap (*x)
--         hv = scale p hv0 + scale q hw0
--         hw = scale q hv0 + scale p hw0
--         bia = dirBia ^^. dirBiaff
--      in (bia ^^. biaffV) `dot`
--           (FFN.run (bia ^^. biaffN) (hv # hw))
-- 
-- 
-- ----------------------------------------------
-- ----------------------------------------------
-- 
-- 
-- -- -- | A version of `Biaff` in which the label between two nodes determines who
-- -- -- is the ,,semantic'' parent.
-- -- data DirBiaff dim h b = DirBiaff
-- --   { _dirBiaMap :: M.Map b Double
-- --   , _dirBiaff :: Biaff dim h
-- --   } deriving (Generic, Binary, NFData, ParamSet)
-- -- 
-- -- instance (KnownNat dim, KnownNat h, Ord b) => Backprop (DirBiaff dim h b)
-- -- makeLenses ''DirBiaff
-- -- 
-- -- instance (KnownNat dim, KnownNat h, Ord b) => New a b (DirBiaff dim h b) where
-- --   new nodeLabelSet arcLabelSet = DirBiaff
-- --     -- TODO: could be simplified...
-- --     <$> mkMap arcLabelSet (pure 0)
-- --     <*> new nodeLabelSet arcLabelSet
-- --     where
-- --       mkMap keySet mkVal = fmap M.fromList .
-- --         forM (S.toList keySet) $ \key -> do
-- --           (key,) <$> mkVal
-- -- 
-- -- instance (KnownNat dim, KnownNat h, Ord b, Show b)
-- --   => BiComp dim a b (DirBiaff dim h b) where
-- --   runBiComp graph (v, w) dirBia =
-- --     let nodeMap = fmap
-- --           (BP.constVar . nodeEmb)
-- --           (nodeLabelMap graph)
-- --         hv = nodeMap M.! v
-- --         hw = nodeMap M.! w
-- --         err = trace
-- --           ( "Graph.DirBiaff: unknown arc label ("
-- --           ++ show (M.lookup (v, w) (arcLabelMap graph))
-- --           ++ ")" ) 1
-- --         -- logist x = 1.0 / (1.0 + exp (-100*x))
-- --         logist = logistic
-- --         p = maybe err logist $ do
-- --               arcLabel <- M.lookup (v, w) (arcLabelMap graph)
-- --               dirBia ^^. dirBiaMap ^^? ixAt arcLabel
-- --         q = 1 - p
-- --         bia = dirBia ^^. dirBiaff
-- --         pot1 = (bia ^^. biaffV) `dot`
-- --                   (FFN.run (bia ^^. biaffN) (hv # hw))
-- --         pot2 = (bia ^^. biaffV) `dot`
-- --                   (FFN.run (bia ^^. biaffN) (hw # hv))
-- --      in if p > 0.99
-- --         then p*pot1
-- --         else if q > 0.99
-- --         then q*pot2
-- --         else p*pot1 + q*pot2
-- 
-- 
-- ----------------------------------------------
-- ----------------------------------------------
-- 
-- 
-- -- | Holistic compoment -- (almost) all in one
-- --
-- --   * @d1@ -- word embedding dimension
-- --   * @d2@ -- label embedding dimension
-- --   * @h@  -- hidden dimension
-- --   * @o@  -- pre-output dimension
-- --
-- data Holi d1 d2 a b h o = Holi
--   { _holN :: FFN
--       (d1 Nats.+ d1 Nats.+ d2 Nats.+ d2 Nats.+ d2)
--       -- (d1 Nats.+ d1 Nats.+ d2 Nats.+ d2)
--         -- two words + two POS + dep label
--       h
--         -- hidden layer size
--       o
--         -- output layer size
--   , _holV :: R o
--       -- ^ To get the ultimate potential value via dot product
--   , _holHeadPosMap :: M.Map a (R d2)
--       -- ^ Head POS repr
--   , _holDepPosMap :: M.Map a (R d2)
--       -- ^ Dependent POS repr
--   , _holArcMap :: M.Map b (R d2)
--       -- ^ Arc label repr
--   } deriving (Generic, Binary, NFData, ParamSet)
-- 
-- instance (KnownNat d1, KnownNat d2, Ord a, Ord b, KnownNat h, KnownNat o)
--   => Backprop (Holi d1 d2 a b h o)
-- makeLenses ''Holi
-- 
-- instance (KnownNat d1, KnownNat d2, Ord a, Ord b, KnownNat h, KnownNat o)
--   => New a b (Holi d1 d2 a b h o) where
--   new xs ys = Holi
--     <$> new xs ys
--     <*> new xs ys
--     <*> newMap xs xs ys
--     <*> newMap xs xs ys
--     <*> newMap ys xs ys
-- 
-- instance
--   ( KnownNat d1, KnownNat d2, KnownNat h, KnownNat o
--   , Show a, Ord a, Show b, Ord b )
--   => BiComp d1 a b (Holi d1 d2 a b h o) where
--     runBiComp graph (v, w) holi =
--       let nodeMap = nodeLabelMap graph
--           wordMap = fmap (BP.constVar . nodeEmb) nodeMap
--           hv = wordMap M.! v
--           hw = wordMap M.! w
--           depPos = depPosRepr . nodeLab $ nodeMap M.! v
--           headPos = headPosRepr . nodeLab $ nodeMap M.! w
--           arc = arcRepr $ arcLabelMap graph M.! (v, w)
--        in (holi ^^. holV) `dot`
--             (FFN.run (holi ^^. holN) (hv # hw # depPos # headPos # arc))
--       where
--         headPosRepr pos = maybe err id $ do
--           holi ^^. holHeadPosMap ^^? ixAt pos
--           where
--             err = trace
--               ( "Graph2.BiComp: unknown POS ("
--               ++ show pos
--               ++ ")" ) 0
--         depPosRepr pos = maybe err id $ do
--           holi ^^. holDepPosMap ^^? ixAt pos
--           where
--             err = trace
--               ( "Graph2.BiComp: unknown POS ("
--               ++ show pos
--               ++ ")" ) 0
--         arcRepr dep = maybe err id $ do
--           holi ^^. holArcMap ^^? ixAt dep
--           where
--             err = trace
--               ( "Graph2.BiComp: unknown arc label ("
--               ++ show dep
--               ++ ")" ) 0
