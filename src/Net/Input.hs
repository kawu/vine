{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- {-# LANGUAGE Rank2Types #-}
-- {-# LANGUAGE LambdaCase #-}
-- {-# LANGUAGE UndecidableInstances #-}
-- {-# LANGUAGE AllowAmbiguousTypes #-}
-- {-# LANGUAGE InstanceSigs #-}

-- {-# LANGUAGE MultiParamTypeClasses #-}
-- -- {-# LANGUAGE RecordWildCards #-}
-- -- {-# LANGUAGE PatternSynonyms #-}
-- -- {-# LANGUAGE ViewPatterns #-}
-- {-# LANGUAGE FlexibleContexts #-}
-- {-# LANGUAGE FlexibleInstances #-}
-- {-# LANGUAGE TupleSections #-}
-- {-# LANGUAGE DataKinds #-}
-- {-# LANGUAGE DeriveFunctor #-}
-- {-# LANGUAGE DeriveFoldable #-}
-- -- {-# LANGUAGE DeriveTraversable #-}
--
-- {-# LANGUAGE PolyKinds #-}
-- {-# LANGUAGE NoMonomorphismRestriction #-}

-------------------------
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
-- {-# OPTIONS_GHC -fconstraint-solver-iterations=3 #-}
-------------------------


-- | Input transformation layer


module Net.Input
  ( -- * Input
    Input(..)
  , evalInput

  , RawInp(..)
  , PosDepInp(..)

    -- * Transform
  , Transform(..)
  , evalTransform

  , NoTrans(..)
  , Scale(..)
  , ScaleLeakyRelu(..)
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat)
import qualified GHC.TypeNats as Nats

import           Control.DeepSeq (NFData)
import           Control.Lens.At (ixAt)

import           Lens.Micro.TH (makeLenses)

import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, Reifies, W, BVar, (^^.), (^^?))
import           Numeric.LinearAlgebra.Static.Backprop (R, L, (#), (#>)) -- dot
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP

import           Data.Binary (Binary)
import qualified Data.Map.Strict as M
import qualified Data.Text as T

import           Numeric.SGD.ParamSet (ParamSet)

import           Net.Pair
import           Net.New
import qualified Net.Util as U
import qualified Format.Cupt as Cupt

import           Debug.Trace (trace)


----------------------------------------------
-- Classes
----------------------------------------------


-- | Input extraction layer
class (KnownNat i, KnownNat o, Backprop bp) => Input bp i o where
  runInput
    :: (Reifies s W)
    => BVar s bp
      -- ^ Network parameters (over which we can backpropagate)
    -> [(Cupt.Token, R i)]
      -- ^ Input sentence with the corresponding embeddings.  WARNING: the
      -- function must not rely on `Cupt.mwe` annotations.
    -> [BVar s (R o)]


evalInput
  :: forall bp i o. (Input bp i o)
  => bp
    -- ^ Network parameters (over which we can backpropagate)
  -> [(Cupt.Token, R i)]
    -- ^ Input sentence with the corresponding embeddings.  WARNING: the
    -- function must not rely on `Cupt.mwe` annotations.
  -> [R o]
evalInput bp0 xs =
  BP.evalBP run bp0
  where
    run :: forall s. (Reifies s W) => BVar s bp -> BVar s [R o]
    run bp = BP.collectVar (runInput bp xs)


-- instance (Input bp1 i o1, Input bp2 i o2, KnownNat o, o ~ (o1 Nats.+ o2))
--   => Input (bp1 :& bp2) i o where
--   -- => Input (bp1 :& bp2) i o where
--   runInput (bp1 :&& bp2) = undefined


-- | Input transformation layer
class (KnownNat i, KnownNat o, Backprop bp) => Transform bp i o where
  runTransform
    :: (Reifies s W)
    => BVar s bp
    -> [BVar s (R i)]
    -> [BVar s (R o)]


evalTransform
  :: forall bp i o. (Transform bp i o)
  => bp
  -> [R i]
  -> [R o]
evalTransform bp0 xs =
  BP.evalBP run bp0
  where
    run :: forall s. (Reifies s W) => BVar s bp -> BVar s [R o]
    run bp = BP.collectVar . runTransform bp $ map BP.auto xs


-- instance (Transform bp1 i k, Transform bp2 k o)
--   => Transform (bp1 :& bp2) i o where
--   -- runTransform :: (Reifies s W) => BVar s bp -> [BVar s (R i)] -> [BVar s (R o)]
--   runTransform (bp1 :&& bp2) xs =
--     runTransform bp2 (runTransform bp1 xs) -- :: [BVar s (R k)])
--     -- runTransform bp2 xs


----------------------------------------------
----------------------------------------------


-- | Raw input layer which only perserves the embeddings.
data RawInp = RawInp
  deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)

instance (a ~ T.Text, b ~ T.Text) => New a b RawInp where
  new _xs _ys = pure RawInp

instance (KnownNat i, i ~ j) => Input RawInp i j where
  runInput _ sent = do
    (_tok, emb)  <- sent
    return (BP.auto emb)


----------------------------------------------
----------------------------------------------


-- | Input layer which concatenates the input embeddings with POS and DEP
-- representations.
data PosDepInp p d = PosDepInp
  { _posMap :: M.Map T.Text (R p)
    -- ^ POS representation
  , _depMap :: M.Map T.Text (R d)
    -- ^ Deprel representation
  } deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)

makeLenses ''PosDepInp

instance (KnownNat p, KnownNat d, a ~ T.Text, b ~ T.Text)
  => New a b (PosDepInp p d) where
  new xs ys = PosDepInp
    <$> newMap xs xs ys
    <*> newMap ys xs ys

instance 
  ( KnownNat i, KnownNat p
  , KnownNat d, KnownNat o
  , o ~ (i Nats.+ p Nats.+ d)
  ) =>
  Input (PosDepInp p d) i o where
  runInput pd sent = do

    (tok, emb)  <- sent
    let pos = posRepr (Cupt.upos tok)
        dep = depRepr (Cupt.deprel tok)

    return (BP.auto emb # pos # dep)

    where
  
      posRepr pos = maybe err id $ do
        pd ^^. posMap ^^? ixAt pos
        where
          err = trace
            ( "Graph2.BiComp: unknown POS ("
            ++ show pos
            ++ ")" ) 0
  
      depRepr dep = maybe err id $ do
        pd ^^. depMap ^^? ixAt dep
        where
          err = trace
            ( "Graph2.BiComp: unknown arc label ("
            ++ show dep
            ++ ")" ) 0


----------------------------------------------
----------------------------------------------
     
    
-- | No transformation
data NoTrans = NoTrans
  deriving (Show, Generic, Binary, NFData, ParamSet, Num, Backprop)

makeLenses ''NoTrans

instance New a b NoTrans where
  new xs ys = pure NoTrans

instance (KnownNat i, i ~ o) => Transform NoTrans i o where
  runTransform _ = id


----------------------------------------------
----------------------------------------------
     
    
-- | Simple scaling layer which allows to scale down the size of vector
-- representations.
newtype Scale i o = Scale
  { _contractL :: L o i
  } deriving (Show, Generic)
    deriving newtype (Binary, NFData, ParamSet, Num, Backprop)
    -- TODO: why do we need the `Num` instance?

makeLenses ''Scale

instance (KnownNat i, KnownNat o) => New a b (Scale i o) where
  new xs ys = Scale <$> new xs ys

instance (KnownNat i, KnownNat o, i ~ i', o ~ o')
  => Transform (Scale i o) i' o' where
  runTransform bp = map (bp ^^. contractL #>)


-- | Simple scaling layer which allows to scale down the size of vector
-- representations.
newtype ScaleLeakyRelu i o = ScaleLeakyRelu
  { _contractLRL :: L o i
  } deriving (Show, Generic)
    deriving newtype (Binary, NFData, ParamSet, Num, Backprop)

makeLenses ''ScaleLeakyRelu

instance (KnownNat i, KnownNat o) => New a b (ScaleLeakyRelu i o) where
  new xs ys = ScaleLeakyRelu <$> new xs ys

instance (KnownNat i, KnownNat o, i ~ i', o ~ o')
  => Transform (ScaleLeakyRelu i o) i' o' where
  runTransform bp = map (U.leakyRelu . (bp ^^. contractLRL #>))
  -- runTransform bp = map (LBP.vmap' U.leakyRelu . (bp ^^. contractLRL #>))


----------------------------------------------
-- Activation functions
----------------------------------------------
     
    
data Relu = Relu
  deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)

instance (KnownNat i) => Transform Relu i i where
  -- runTransform _ = map (LBP.vmap' U.relu)
  runTransform _ = map U.relu


data LeakyRelu = LeakyRelu
  deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)

instance (KnownNat i) => Transform LeakyRelu i i where
  -- runTransform _ = map (LBP.vmap' U.leakyRelu)
  runTransform _ = map U.leakyRelu


data Logistic = Logistic
  deriving (Show, Generic, Binary, NFData, ParamSet, Backprop)

instance (KnownNat i) => Transform Logistic i i where
  runTransform _ = map U.logistic
